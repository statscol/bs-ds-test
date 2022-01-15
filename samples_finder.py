from utils import *
from config import MODEL_NAME_HF
from sentence_transformers import SentenceTransformer
import argparse

PRO_SCORES=pd.read_pickle('pro_scores.pkl')
SAMPLES_COMPOSED_DATA=pd.read_pickle('samples_composed_data.pkl')


class SearchRecommender:
    
    def __init__(self):
        
        self.samples_embeddings=SAMPLES_COMPOSED_DATA
        self.pros=PRO_SCORES
        self.model=SentenceTransformer(MODEL_NAME_HF)

    @staticmethod
    def detect_intent(search: str):
        """
        future development to detect category in a search pattern
        """
        pass

    def generate_similarities(self,search: str):
        """
        computes cosine similarity of search and every sample attributes and tags using its embeddings
        @param search: str | pattern to search for samples
        """
        search=decode_language_code(search,sep=" ")
        search=clean_sentence(search)
        search_emb=self.model.encode(search)
        self.samples_embeddings.loc[:,'similarity']=self.samples_embeddings.avg_embeddings.apply(lambda d: cosine_sim(d,search_emb))
        print('[INFO] Similarities created')
    
    def get_suggestions(self,search: str="uk voice over"):
        """
        generates suggestions list given a search pattern

        """  
        search_clean=clean_sentence(search)
        self.generate_similarities(search=search_clean)
        return self.samples_embeddings.drop(columns=['avg_embeddings','no_tag']).sort_values(by="similarity",ascending=False).merge(self.pros[['pro_id','performance_score']],on="pro_id",how="inner")

    
    @staticmethod
    def get_batch_suggestions(df,by_batch_size: int=50):
        pass


if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Search Recommender v.0.0.1")
    parser.add_argument("-s","--search",help="Search String",dest="search_pattern",type=str)
    args=parser.parse_args()
    print(args.search_pattern)
    
    finder=SearchRecommender()
    results=finder.get_suggestions(search=args.search_pattern)
    print(results.shape)
    print(results.head())