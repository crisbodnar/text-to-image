class GanClsTrainer(object):
    def __init__(config):
        return

    def train(self, config):
        D_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
            .minimize(self.D_loss, var_list=self.d_vars)
        G_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
            .minimize(self.G_loss, var_list=self.g_vars)

        tf.global_variables_initializer().run()

        self.G_merged_summ = merge_summary([self.z_sum, self.G_summ])
        self.D_merged_summ = merge_summary([self.z_sum, self.D_real_mismatch_summ, self.D_real_match_summ,
                                           self.D_synthetic_summ])
        self.writer = SummaryWriter("./logs", self.sess.graph)

        # TODO: There is a bug which enforces the sample num to be the bath size.
        sample_z = np.random.uniform(-1, 1, size=(self.sample_num, self.z_dim))
        _, sample_phi, _, captions = self.dataset.test.next_batch_test(self.sample_num,
                                                                       randint(0, self.dataset.test.num_examples), 1)
        sample_phi = np.squeeze(sample_phi, axis=0)
        print(sample_phi.shape)

        # Display the captions of the sampled images
        print('\nCaptions of the sampled images:')
        for caption_idx, caption_batch in enumerate(captions):
            print('{}: {}'.format(caption_idx + 1, caption_batch[0]))
        print()

        counter = 1
        start_time = time.time()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for epoch in range(config.epoch):
            # Updates per epoch are given by the training data size / batch size
            updates_per_epoch = self.dataset.train.num_examples // self.batch_size

            for idx in range(0, updates_per_epoch):
                images, wrong_images, phi, _, _ = self.dataset.train.next_batch(self.batch_size, 4)

                batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]).astype(np.float32)

                # Update D network
                _, err_d_real_match, err_d_real_mismatch, err_d_fake, err_d, summary_str = self.sess.run(
                    [D_optim, self.D_real_match_loss, self.D_real_mismatch_loss, self.D_synthetic_loss,
                     self.D_loss, self.D_merged_summ],
                    feed_dict={
                        self.inputs: images,
                        self.wrong_inputs: wrong_images,
                        self.phi_inputs: phi,
                        self.z: batch_z
                    })
                self.writer.add_summary(summary_str, counter)

                # Update G network
                _, err_g, summary_str = self.sess.run([G_optim, self.G_loss, self.G_merged_summ],
                                                      feed_dict={self.z: batch_z, self.phi_inputs: phi})
                self.writer.add_summary(summary_str, counter)

                # # Run G_optim twice to make sure that d_loss does not go to zero (different from paper)
                # _, err_g, summary_str = self.sess.run([G_optim, self.G_loss, self.G_merged_summ],
                #                                       feed_dict={self.z: batch_z, self.phi_inputs: phi})
                # self.writer.add_summary(summary_str, counter)

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, idx, updates_per_epoch,
                         time.time() - start_time, err_d, err_g))

                if np.mod(counter, 100) == 1:
                    try:
                        samples = self.sess.run(self.sampler, feed_dict={self.z_sample: sample_z,
                                                                         self.phi_sample: sample_phi,
                                                                         })
                        save_images(samples, image_manifold_size(samples.shape[0]),
                                    './{}/{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, 'GANCLS', epoch, idx))
                        print("[Sample] d_loss: %.8f, g_loss: %.8f" % (err_d, err_g))
                    except Exception as excep:
                        print("one pic error!...")
                        print(excep)

                if np.mod(counter, 500) == 2:
                    self.save(config.checkpoint_dir, counter)