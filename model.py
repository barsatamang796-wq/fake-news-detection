from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

# TRAINING DATA
news = [

    # REAL NEWS
    "Government launches new education scheme",
    "India wins cricket world cup",
    "Stock market reaches new high",
    "Scientists discover vaccine",
    "Railway project announced",
    "New hospital opened in Chennai",
    "NASA launches satellite successfully",
    "Weather department predicts heavy rain",
    "Election results announced officially",
    "Researchers develop new AI system",
    "Police arrest robbery suspects",
    "International cricket match scheduled tomorrow",
    "Supreme Court passes important judgement",
    "New metro project starts in city",
    "Doctors discover treatment for disease",
    "Technology company launches smartphone",
    "Education minister announces scholarships",
    "Heavy rainfall expected in Tamil Nadu",
    "Scientists publish climate research",
    "Government increases public safety measures",

    # FAKE NEWS
    "Aliens attacked Earth last night",
    "Dinosaurs found alive in Chennai",
    "Zombie outbreak reported worldwide",
    "Humans can fly naturally",
    "Time travel machine sold online",
    "Scientists confirm dragons are real",
    "Moon disappeared from the sky",
    "Invisible humans spotted in city",
    "People can survive without oxygen forever",
    "Sun will not rise tomorrow",
    "Magic water cures all diseases instantly",
    "Ghost elected as president",
    "Animals started speaking English",
    "Earth will explode tonight",
    "Secret portal to Mars discovered",
    "Flying cars powered by magic launched",
    "Immortality pill available online",
    "Mermaids found in Indian Ocean",
    "Dead people brought back to life",
    "Alien school opened on Moon"
]

labels = [

    # REAL
    "REAL","REAL","REAL","REAL","REAL",
    "REAL","REAL","REAL","REAL","REAL",
    "REAL","REAL","REAL","REAL","REAL",
    "REAL","REAL","REAL","REAL","REAL",

    # FAKE
    "FAKE","FAKE","FAKE","FAKE","FAKE",
    "FAKE","FAKE","FAKE","FAKE","FAKE",
    "FAKE","FAKE","FAKE","FAKE","FAKE",
    "FAKE","FAKE","FAKE","FAKE","FAKE"
]

# VECTORIZATION
vectorizer = TfidfVectorizer(stop_words="english")

X = vectorizer.fit_transform(news)

# MODEL
model = PassiveAggressiveClassifier(max_iter=1000)

model.fit(X, labels)

# PREDICTION FUNCTION
def predict_news(text):

    # CUSTOM RULES
    fake_keywords = [

        "alien",
        "zombie",
        "dragon",
        "ghost",
        "magic",
        "immortality",
        "flying human",
        "time travel",
        "mermaid"
    ]

    text_lower = text.lower()

    for word in fake_keywords:

        if word in text_lower:

            return "FAKE"

    # ML PREDICTION
    vec = vectorizer.transform([text])

    prediction = model.predict(vec)[0]

    return prediction