class DataBatch(object):
    def __init__(self):
        self.class_ids = None
        self.images = None
        self.wrong_images = None
        self.embeddings = None
        self.wrong_embeddings = None
        self.captions = None
        self.labels = None
        self.ids = None
