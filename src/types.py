from typing import TypedDict, List, Tuple

from src.subgraphrag.subgraph import Subgraph


class TagMeAnnotation(TypedDict, total=False):
    spot: str
    start: int
    end: int
    id: int
    title: str
    rho: float
    link_probability: float
    abstract: str
    dbpedia_categories: List[str]


class RawTrainQuestion(TypedDict):
    questionid: str
    utterance: str
    answers: List[str | None]
    answers_str: List[str | None]

class CleanedQuestion(TypedDict):
    id: str
    question: str
    answer_entities: List[str]

class AnnotatedQuestion(TypedDict):
    id: str
    question: str
    answer_entities: List[str]
    annotations: List[TagMeAnnotation]

class QuestionWithEntityTitles(TypedDict):
    id: str
    question: str
    answer_entities: List[str]
    titles: List[str]

class SRTKStructure(TypedDict):
    id: str
    question: str
    question_entities: List[str]
    answer_entities: List[str]

class SRTKStructureWithTriplets(SRTKStructure):
    triplets: List[List[str]]


class ContextGenerationStructure:
    id: str
    question: str
    subgraphs: List[Subgraph]

class SRTKStructureWithAnswerStrings(SRTKStructure):
    answers_str: List[str]