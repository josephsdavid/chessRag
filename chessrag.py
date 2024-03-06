import llama_index
from llama_index.llms.huggingface import HuggingFaceLLM
import numpy as np
import chess
from io import StringIO
from chess import pgn
import chromadb
import uuid
from uuid import UUID, uuid5
import tqdm
import re
from datetime import datetime, timedelta

def generate_dates(start_date_str):
    # Convert the input start date string to a datetime object
    start_date = datetime.strptime(start_date_str, '%Y.%m.%d')
    # Get today's date
    end_date = datetime.now()
    # Generate a list of dates from start_date to end_date
    date_list = [(start_date + timedelta(days=x)).strftime('%Y.%m.%d') for x in range((end_date - start_date).days + 1)]
    return date_list

#
# Useful consts
#

FORCE = False
NAMESPACE = UUID('6ef7b608-8ef8-4c30-95a4-2967f4ce0976')
PGN_STORE = "data/base.pgn"
HEADER_KEYS = [
    "Date", "Site", "White", "Black", "Result", "Round", "ECO", "Annotator"
]

#
# Chess utilities
#

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


def read_pgn_from_string(pgn_string):
    pgn_io = StringIO(pgn_string)
    game = chess.pgn.read_game(pgn_io)
    return game


#
# Tools for creating vector dbs
#

def id_from_headers(headers):
    return str(uuid5(NAMESPACE, str(dict(headers))))


def add_game_and_headers_to_collection(collection, game, headers):
    collection.add(documents=[str(game)],
                   metadatas=[dict(headers)],
                   ids=[id_from_headers(headers)])


def add_game_to_collection(collection, game):
    headers = game.headers
    add_game_and_headers_to_collection(collection, game, headers)


#
# Stripping variations, comments, notes, evaluations, etc.
#
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


# a helper for querying collections because im lazy
def query_collection_with_game(collection, game, n_results, **kwargs):
    return collection.query(query_texts=[game], n_results=n_results, **kwargs)


#
# Similarity between games!
#

# for now, this is based entirely off of pawn structure,
# if the same (non initial) pawn structure occurs in a game,
# odds are at some point something similar happened in both games
def extract_pawn_structure(game):
    pawn_structures = []
    board = chess.Board()

    for move in game.mainline_moves():
        board.push(move)
        pawn_structure = board.board_fen().split(' ')[0]
        pawn_structures.append(pawn_structure)

    return pawn_structures


# calculates jaccard similarity for all pawn structures
def pawn_structure_similarity(game1, game2):
    pawn_structures1 = extract_pawn_structure(game1)
    pawn_structures2 = extract_pawn_structure(game2)

    all_similarities = []
    for pawn1 in set(pawn_structures1):
        for pawn2 in set(pawn_structures2):
            __import__('pdb').set_trace()
            intersection = len(set(pawn1).intersection(set(pawn2)))
            union = len(set(pawn1).union(set(pawn2)))
            similarity = intersection / union if union > 0 else 0.0
            all_similarities.append(similarity)

    return sum(all_similarities) / len(all_similarities)

#
# Prompting, and doing the actual LLM bit (see joaquin example more)
#
def generate_prompt(prompt_documents, game_string):

    prompt_documents = " __NEW_GAME__ ".join(prompt_documents)
    return f"""
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

    assert len(closest_matches["documents"]) == 1 == len(closest_matches["ids"])
    assert len(closest_matches["documents"][0]) == 100 == len(closest_matches["ids"][0])
    similarities = [pawn_structure_similarity(game, read_pgn_from_string(match_game)) for match_game in tqdm.tqdm(closest_matches["documents"][0])]
    indices = np.array(sorted(range(len(similarities)), key=lambda idx: similarities[idx], reverse=True))
    ids = np.array(closest_matches["ids"][0])
    ranked_ids = ids[indices][:5].tolist()
    best_games = annotated.get(ranked_ids)["documents"]
    prompt = generate_prompt(best_games, str(game))

    # XXX: Almost done, However my prompt is extremely long! How do i deal with this? How does chunking work?
    llm = HuggingFaceLLM(
        model_name="HuggingFaceH4/zephyr-7b-beta",
        tokenizer_name="HuggingFaceH4/zephyr-7b-beta",
        context_window=3900,
        max_new_tokens=256,
        # model_kwargs={"quantization_config": quantization_config},
        # tokenizer_kwargs={},
        generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95},
        device_map="auto",
    )
    __import__('pdb').set_trace()
