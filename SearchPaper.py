from sentence_transformers import SentenceTransformer, util
import json
path = "all-MiniLM-L6-v2"
model = SentenceTransformer(path)

def myredect(mystr):
    print(mystr)
    sentences = []
    returnscores = []
    returnsentences = []
    sentences.append(mystr)
    with open("EMNLP16.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
        if isinstance(data, list):
            for item in data:
                if 'title' in item:
                    sentences.append(item['title'])

        # Compute embeddings
        embeddings = model.encode(sentences, convert_to_tensor=True)

        # Compute cosine-similarities for each sentence with each other sentence
        cosine_scores = util.cos_sim(embeddings, embeddings)

        pairs = []
        for i in range(cosine_scores.shape[0]):
            for j in range(cosine_scores.shape[1]):
                pairs.append({"index": [i, j], "score": cosine_scores[i][j]})
        pairs = sorted(pairs, key=lambda x: x["score"], reverse=True)
        length = len(pairs)
        k = 0
        for pair in pairs:
            i, j = pair["index"]
            if abs(pair["score"].item() - 1.000) < 0.00001:
                continue
            if sentences[i] == mystr:
                returnsentences.append(sentences[j])
                returnscores.append(pair["score"].item())
                k += 1
                # print("{} \t\t {} \t\t Score: {:.4f}".format(
                #     sentences[i], sentences[j], pair["score"]
                # ))
            if k == 10:
                break


        return returnsentences,returnscores

