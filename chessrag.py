import llama_index
import chess
from chess import pgn
import chromadb
import uuid
from uuid import UUID, uuid5
import tqdm
import re

# setup Chroma in-memory, for easy prototyping. Can add persistence easily!

FORCE = False
NAMESPACE = UUID('6ef7b608-8ef8-4c30-95a4-2967f4ce0976')
PGN_STORE = "data/base.pgn"
HEADER_KEYS = [
    "Date", "Site", "White", "Black", "Result", "Round", "ECO", "Annotator"
]


# python chess library is annoying, add a convenience wrapper
class PgnLoader:

    def __init__(self, store):
        f = open(store)
        offsets = []
        while True:
            offset = f.tell()
            headers = pgn.read_headers(f)
            if headers is None:
                break
            offsets.append(offset)
        self.pgns = f
        self.offsets = offsets

    def __getitem__(self, item):
        offset = self.offsets[item]
        self.pgns.seek(offset)
        return pgn.read_game(self.pgns)

    def __len__(self):
        return len(self.offsets)


def id_from_headers(headers):
    return str(uuid5(NAMESPACE, str(dict(headers))))


def add_game_and_headers_to_collection(collection, game, headers):
    collection.add(documents=[str(game)],
                   metadatas=[dict(headers)],
                   ids=[id_from_headers(headers)])


def add_game_to_collection(collection, game):
    headers = game.headers
    add_game_and_headers_to_collection(collection, game, headers)


def process_pgn(input_string):
    # Remove characters after dollar signs and up to the next space
    dollar_removed = re.sub(r'\$[^\s]*', '', input_string)

    # Remove text within squiggly brackets
    brackets_removed = re.sub(r'\{[^}]*\}', '', dollar_removed)

    # Remove all non-alphanumeric characters
    alphanumeric_only = re.sub(r'[^a-zA-Z0-9\s]', '', brackets_removed)

    return alphanumeric_only


def process_chessgame(game):
    return process_pgn(str(game))


def query_collection_with_game(collection, game, n_results, **kwargs):
    return collection.query(query_texts=[game], n_results=n_results, **kwargs)


def extract_pawn_structure(game):
    pawn_structures = []
    board = chess.Board()

    for move in game.mainline_moves():
        board.push(move)
        pawn_structure = board.board_fen().split(' ')[0]
        pawn_structures.append(pawn_structure)

    return pawn_structures

def pawn_structure_similarity(game1, game2):
    pawn_structures1 = extract_pawn_structure(game1)
    pawn_structures2 = extract_pawn_structure(game2)

    # Calculate the Jaccard similarity between sets of pawn structures
    intersection = len(set(pawn_structures1).intersection(set(pawn_structures2)))
    union = len(set(pawn_structures1).union(set(pawn_structures2)))

    similarity = intersection / union if union > 0 else 0.0
    return similarity

def generate_prompt(prompt_documents, game_string):

    prompt_documents = " __NEW_GAME__ ".join(prompt_documents)
    prompt = f"""
    You are an automated chess coach. Your job is to annotate a chess game for your student,
    providing:
    helpful comments in {{}} squiggly brackets,
    New variations in () parentheses,
    evaluations in the following form:
    `=` means the position is equal
    `+=` means the position is slightly better for white
    `+/-` means the position is better for white
    `+-` means the position is completely winning for white
    `~` means the position is unclear
    `=+` means the position is slightly better for black
    `-/+` means the position is better for black
    `-+` means the position is completely winning for black
    you can also annotate moves as follows:
    `!?` is an interesting move
    `?!` is a dubious move
    `?` is a mistake
    `??` is a blunder (an instantly losing error)
    `!` is an excellent move (only given out for difficult to find moves)
    `!!` is for a brilliant move (almost never given out, only for exceptional, often counterintuitive moves)

    You can ignore any information inside of [] normal brackets. For your help, we have already annotated a few games for you below.
    Games are split by the phrase " __NEW_GAME__ ":


    {prompt_documents}

    Now it is your turn. Please annotate the following game and return the output:

    {game_string}
    """





# collection.get(document_id) to query by id

if __name__ == "__main__":
    loader = PgnLoader(PGN_STORE)
    client = chromadb.PersistentClient(path="vectors/chess")
    annotated = client.get_or_create_collection("chesspub")
    raw_pgns = client.get_or_create_collection("chesspub_pgn")

    if FORCE:
        client.delete_collection(name="chesspub")
        client.delete_collection(name="chesspub_pgn")
        annotated = client.create_collection("chesspub")
        raw_pgns = client.get_or_create_collection("chesspub_pgn")
        # Create annotated. get_collection, get_or_create_collection, delete_collection also available!
        for game in tqdm.tqdm(loader):
            add_game_to_collection(annotated, game)
            add_game_and_headers_to_collection(
                raw_pgns, process_chessgame(game.mainline_moves()),
                game.headers)

    with open("data/g1.pgn") as f:
        game = pgn.read_game(f)


    closest_matches = query_collection_with_game(
        raw_pgns,
        process_chessgame(game.mainline_moves()),
        n_results=100
    )
    ids = closest_matches["ids"][0]
    # TODO: Make reranker
    # TODO: Pass prompt to LLM
    # TODO: Output
    __import__('pdb').set_trace()
