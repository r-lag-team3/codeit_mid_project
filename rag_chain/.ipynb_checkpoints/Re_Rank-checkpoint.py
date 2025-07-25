{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "58572886-f247-491b-8b19-23dc78ef9ed8",
   "metadata": {},
   "outputs": [],
   "source": [
    "def rerank_documents(question, documents):\n",
    "    pairs = [(question, doc.page_content) for doc in documents]\n",
    "\n",
    "    inputs = tokenizer(\n",
    "        [q for q, p in pairs],\n",
    "        [p for q, p in pairs],\n",
    "        padding=True,\n",
    "        truncation=True,\n",
    "        return_tensors=\"pt\"\n",
    "    )\n",
    "    with torch.no_grad():\n",
    "        outputs = reranker_model(**inputs)\n",
    "        scores = torch.softmax(outputs.logits, dim=1)[:, 1]\n",
    "\n",
    "    # 정렬\n",
    "    reranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)\n",
    "\n",
    "    # 문서만 추출\n",
    "    return [doc for doc, score in reranked]"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python (myenv)",
   "language": "python",
   "name": "myenv"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
