from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken
import openai


def embedding_cosine_similarity(references, candidates, model = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
    """
    Calculates the cosine value (taking values ​​between 0 and 1) of the embeddings for each pair of sentences,
    with one of them as a reference and another as a candidate.
    INPUTS:
        - references: list containing all reference phrases
        - candidates: list containing all candidate phrases
        - model: embeddings model considered (default: 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    OUTPUT
        - list of all the metric values, one for each pair of sentences.
    """
    model = SentenceTransformer(model)
    arr = []
    for i in range(len(references)):
        sentences = [references[i], candidates[i]]
        embeddings = model.encode(sentences)
        arr.append(cosine_similarity(embeddings)[0,1])
    return arr


def bertscores(references, candidates, encoding = tiktoken.encoding_for_model("gpt-3.5-turbo"), model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    """
    Calculates the BERTScore metric for each pair of sentences, with one of them as a reference and another as a candidate, with the difference that any possible embedding model is considered.
    INPUTS:
        - references: list containing all reference phrases
        - candidates: list containing all candidate phrases
        - model: embeddings model considered (default: 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    OUTPUT
        - list of all the metric values, one for each pair of sentences.
    """
    model = SentenceTransformer(model)
    arr = []
    for i in range(len(references)):
        reference = references[i]
        candidate = candidates[i]
        cod1 = encoding.encode(reference)
        dec1 = [encoding.decode([element]) for element in cod1]
        cod2 = encoding.encode(candidate)
        dec2 = [encoding.decode([element]) for element in cod2]

        embeddings1 = model.encode(dec1)
        embeddings2 = model.encode(dec2)
        matrix = cosine_similarity(embeddings1,embeddings2)
        max_by_row = matrix.max(axis=1)
        average = max_by_row.sum()/len(max_by_row)

        arr.append(average)
    return arr

def answer_token_recall_precision(actual_responses, generated_responses):
    """
    Calculates token-level recall and precision between actual responses and generated responses.
    Recall measures the proportion of tokens in the actual response that are also found in the generated response,
    while precision measures the proportion of tokens in the generated response that are found in the actual response.
    
    INPUTS:
        - actual_responses: list of strings containing the actual (reference) responses.
        - generated_responses: list of strings containing the generated responses to be evaluated.
    
    OUTPUT:
        - recalls: list of recall values for each pair of actual and generated response.
        - precisions: list of precision values for each pair of actual and generated response.
    
    Note: This function assumes that both input lists are of the same length and pairs of responses are aligned
    in the order they appear in the lists. A ValueError is raised if the lists have different lengths.
    """
    llm_evaluation_encoding = tiktoken.get_encoding("cl100k_base")
    recalls = []
    precisions = []
    
    # Ensure both lists have the same length
    if len(actual_responses) != len(generated_responses):
        raise ValueError("Lists actual_responses and generated_responses must have the same length.")
    
    # Iterate over pairs of actual and generated responses
    for actual_response, generated_response in zip(actual_responses, generated_responses):

        # Tokenize context and response
        tokens_actual_response = llm_evaluation_encoding.encode(actual_response)
        tokens_generated_response = llm_evaluation_encoding.encode(generated_response)
        
        # Count how many tokens from the actual response appear in the generated response
        hits = sum(token in tokens_generated_response for token in tokens_actual_response)
        
        # Calculate hit ratio for recall and precision
        recall = hits / len(tokens_actual_response) if tokens_actual_response else 0
        precision = hits / len(tokens_generated_response) if tokens_generated_response else 0
        
        recalls.append(recall)
        precisions.append(precision)
    
    return recalls, precisions


def gpt_classify(questions, references, candidates, key):
    """
    It is a classification of a couple of sentences according to GPT3.5. Allows classification as "Correct" or "Incorrect"
    INPUTS:
        - questions: list containing all questions
        - references: list containing all actual reference answers
        - candidates: list containing all generated responses that are candidates
        - key: OpenAI api-key
    OUTPUT
        - list of all classifications, one for each pair of sentences.
    """
    arr = []
    openai_client = openai.OpenAI(
        api_key=key,  # this is also the default, it can be omitted
    )
    for i in range(len(references)):
        question = questions[i]
        reference = references[i]
        candidate = candidates[i]
        prompt_es = f"""Tarea: Evaluación Semántica de la Precisión de Respuestas
                    Contexto: Como asistente analítico, tu responsabilidad es comparar y evaluar respuestas a preguntas específicas. 
                    Se te proporcionará una "Respuesta de Referencia", que se considera la respuesta correcta, y una 
                    "Respuesta Generada" que necesitas evaluar.

                    Instrucciones:
                    1. Pregunta: Lee y comprende la pregunta para identificar el contexto y el tipo de respuesta esperada.
                    2. Respuesta de Referencia: Esta es la respuesta considerada correcta o real para la pregunta.
                    3. Respuesta Generada: Esta es la respuesta que debes evaluar.
                    4. Evaluación Semántica: Basándote en la precisión del contenido y el contexto de la pregunta, evalúa si la "Respuesta Generada" está semánticamente alineada con la "Respuesta de Referencia". Clasifica la "Respuesta Generada" como:
                    - Correcta: Si tiene una alineación semántica con la "Respuesta de Referencia", incluso si incluye información adicional.
                    - Incorrecta: Si no está semánticamente alineada con la "Respuesta de Referencia".

                    Consideraciones Adicionales:
                    - La alineación semántica implica que la esencia, significado o idea principal de las respuestas debe ser la misma, aunque las palabras 
                    o frases utilizadas puedan variar.
                    - Presta atención a los matices y contextos implícitos que podrían afectar la interpretación semántica de las respuestas.

                    Pregunta: {question}
                    Respuesta de referencia: {reference}
                    Respuesta Generada: {candidate}
        """
        prompt = f"""Task: Semantic Evaluation of Response Accuracy
                    Context: As an analytical assistant, your responsibility is to compare and evaluate answers to specific questions.
                    You will be provided with a "Reference Answer", which is considered the correct answer, and a
                    "Generated Response" that you need to evaluate.

                    Instructions:
                    1. Question: Read and understand the question to identify the context and the type of response expected.
                    2. Reference Answer: This is the answer considered correct or real for the question.
                    3. Generated Response: This is the response that you must evaluate.
                    4. Semantic Evaluation: Based on the accuracy of the content and context of the question, evaluate whether the "Generated Response" is semantically aligned with the "Reference Response". Classifies the "Generated Response" as:
                    - Correct: If it has a semantic alignment with the "Reference Response", even if it includes additional information.
                    - Incorrect: If it is not semantically aligned with the "Reference Answer".

                    Additional considerations:
                    - Semantic alignment implies that the essence, meaning or main idea of ​​the answers must be the same, although the words
                    or phrases used may vary.
                    - Pay attention to implicit nuances and contexts that could affect the semantic interpretation of the answers.

                    Question: {question}
                    Reference answer: {reference}
                    Response Generated: {candidate}
        """
        response_openai = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=0
        )
        answer_llm = response_openai.choices[0].message.content
        arr.append(answer_llm)
    return arr