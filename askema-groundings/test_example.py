
import unittest
import logging
import os
import math
import pandas as pd
import shutil
from src.skema_utils import get_cls_embeddings, get_this_triplet_loss, get_triplet_embeddings, init, tokenizer, model
from src.example_triplet_loss import  get_triplet_loss_reults, start

class TestExample(unittest.TestCase):
    def setUp(self) -> None:
        # initialize variables required for tests
        if not tokenizer and not model:
            self.tokenizer, self.model = init()
        else:
            self.tokenizer, self.model = tokenizer, model

        self.root = os.path.dirname(__file__)
        print(os.listdir(os.path.join(self.root, "dataset")))
        self.dataset_path = os.path.join(self.root, "dataset")
        self.name = self.shortDescription() 
        if self.name:
            self.name = "test"
        print(self.name)
        self.files = [ os.path.join(self.dataset_path, file) for file in os.listdir(self.dataset_path)]
        self.file = os.path.join(self.dataset_path, "ged_5Febbuckymodel_webdocs.csv")

        self.fixtures = ['for running simulations','simulation','simulation of exposed organism']

        ## initialize temporary output folders : create now, delete later
        self.output_path = os.path.join(self.root, "temp") 
        print(self.output_path)
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)           

    def test_init(self):
        """Test Initialization"""
        for file in self.files:
            print(file, "\n")
            self.assertTrue(os.path.exists(file))
            df = pd.read_csv(file)
            self.assertEquals(len(self.files), 4)
            self.assertListEqual(df.columns.tolist(),["Unnamed: 0","Text","Context","1","2"] )

        self.assertIsNotNone(tokenizer)
        self.assertIsNotNone(model)
    
    def test_triplet_embeddings(self):
        if os.path.exists(self.file):
            df = pd.read_csv(self.file)
            text, c1, c2 = df["Text"][10],df["1"][10], df["2"][10]
        else:
            #load fixtures
            text, c1, c2 = self.fixtures
        text_embeddings, c1_embeddings, c2_embeddings  = get_triplet_embeddings(tokenizer, model, text, c1, c2, False)
        self.assertListEqual(list(text_embeddings.shape), [1,768])
        self.assertListEqual(list(c1_embeddings.shape), [1,768])
        self.assertListEqual(list(c2_embeddings.shape), [1,768])
    
    def test_triplet_hidden_states_embeddings(self):
        df = pd.read_csv(self.file)
        text, c1, c2 = df["Text"][10],df["1"][10], df["2"][10]
        text_embeddings, c1_embeddings, c2_embeddings  = get_triplet_embeddings(tokenizer, model, text, c1, c2, True)
        self.assertListEqual(list(text_embeddings.shape), [1,5,768])
        self.assertListEqual(list(c1_embeddings.shape), [1,3,768])
        self.assertListEqual(list(c2_embeddings.shape), [1,11,768])  

    def test_list_triplet_embeddings(self):
        df = pd.read_csv(self.file)
        texts, c1, c2 = df["Text"].to_list()[:10],df["1"].to_list()[:10], df["2"].to_list()[:10]
        text_embeddings, c1_embeddings, c2_embeddings = get_triplet_embeddings(tokenizer, model, texts, c1, c2, True)
        self.assertEqual([text_embeddings.shape[0], text_embeddings.shape[2]], [10, 768])
        self.assertListEqual([c1_embeddings.shape[0], c1_embeddings.shape[2] ], [10,768])
        self.assertListEqual([c2_embeddings.shape[0], c2_embeddings.shape[2]], [10,768])

    def test_triplet_cls_embeddings(self):
        df = pd.read_csv(self.file)
        text = df["Text"][10]

        text_cls_embeddings = get_cls_embeddings(tokenizer, model, text, False)
        self.assertListEqual(list(text_cls_embeddings.shape), [1,768])

        text_cls_embeddings = get_cls_embeddings(tokenizer, model, text, True)
        self.assertEqual([text_cls_embeddings.shape[0], text_cls_embeddings.shape[2]], [1, 768])
    
    def test_triplet_loss(self):
        df = pd.read_csv(self.file)
        text, c1, c2 = df["Text"][10],df["1"][10], df["2"][10]
        text_embeddings, c1_embeddings, c2_embeddings  = get_triplet_embeddings(tokenizer, model, text, c1, c2, False)
        loss = get_this_triplet_loss(text_embeddings, c1_embeddings, c2_embeddings)
        self.assertIsNotNone(loss)
        self.assertGreaterEqual(loss, 0.0)

    def test_list_triplet_loss(self):
        df = pd.read_csv(self.file)
        text, c1, c2 = df["Text"].to_list()[:10],df["1"].to_list()[:10], df["2"].to_list()[:10]
        text_embeddings, c1_embeddings, c2_embeddings  = get_triplet_embeddings(tokenizer, model, text, c1, c2, False)
        loss = get_this_triplet_loss(text_embeddings, c1_embeddings, c2_embeddings)
        self.assertIsNotNone(loss)
        self.assertGreaterEqual(loss, 0.0)

    def test_triplet_loss_reults(self):
        df = pd.read_csv(self.file)[:200]
        triplet_loss_dict, easy_triplets, semi_hard_triplets, hard_triplets = get_triplet_loss_reults(df)
        print("Total number of easy, semi-hard and hard triplets respectively for this dataset are : %s \n, %s, %s, %s, %s" 
            %(self.file, str(len(triplet_loss_dict)),str(len(easy_triplets)),str(len(semi_hard_triplets)),str(len(hard_triplets)) ))
        self.assertEqual(len(triplet_loss_dict), 200)
        self.assertEqual(len(easy_triplets), 88)
        self.assertEqual(len(semi_hard_triplets), 22)
        self.assertEqual(len(hard_triplets), 90)
    
    def test_start(self):
        """Test result_files"""
        start(self.output_path, self.files)
        print(self.output_path)
        ## only output files
        files = [os.path.join(self.output_path, file) for file in os.listdir(self.output_path) if not file.startswith("ged")]
        self.assertEqual(len(files), 16)

    def tearDown(self):

        del self.root 
        del self.dataset_path
        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)
        del self.output_path

if __name__ == '__main__':
    unittest.main()

suite = unittest.TestLoader().loadTestsFromTestCase(TestExample)
testResult = unittest.TextTestRunner(verbosity=2).run(suite)