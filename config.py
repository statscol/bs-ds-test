ATTRIBUTES_BY_CATEGORY={
    "audio":['purpose','language_id','gender_and_age_id'],
    "video": ['sample_language','purpose','keywords'],
    "image": ['sample_language',"sample_purpose","sample_service"],
    "article": ['article_language_id','keywords','if_article_specs']
    }

REPLACEMENT_LANGUAGES={
    'eng':'english',
    'spa': 'spanish',
    'fra': 'french',
    'por': 'portuguese',
    'deu': 'german',
    'jpn': 'japanese',
    'kor':'korean',
    'tur':'turkish',
    'dan':'danish',
    'latam':'latino',
    'uk': 'british',
    'fr': 'france',
    'us':'american'
    }


WEIGHTS_SCORE={'samples_rejected_internally':0.4,
                'speed_to_book':0.3,
                'average_review':0.3}

##hugging-face model sentence-transformers
MODEL_NAME_HF="all-MiniLM-L12-v1"

