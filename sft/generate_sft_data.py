import argparse
import json
import math
import chess
import random

PIECE_NAME = {
    "p": "pawn",
    "n": "knight",
    "b": "bishop",
    "r": "rook",
    "q": "queen",
    "k": "king",
}

def get_win_rate(cp, mate):
    # Reference: https://lichess.org/page/accuracy
    # Win% = 50 + 50 * (2 / (1 + exp(-0.00368208 * centipawns)) - 1)
    if mate is not None:
        if mate > 0: return 100.0  
        if mate < 0: return 0.0    
        return 50.0 

    if cp is None: 
        return 50.0

    clamped_cp = max(min(cp, 4000), -4000)
    expiration = math.exp(-0.00368208 * clamped_cp)
    sigmoid = 1 / (1 + expiration)
    
    win_percent = 100 * sigmoid
    
    return round(win_percent, 2)

def generate_candidate_text(fen, candidates, turn_color, shuffle_candidates=False):
    board = chess.Board(fen)
    
    def describe_candidate_move(san, board):
        tmp = board.copy()
        move = tmp.parse_san(san)
        capture_text = ""
        if tmp.is_capture(move):
            if tmp.is_en_passant(move):
                capture_text = "It captures a pawn en passant."
            else:
                captured = tmp.piece_at(move.to_square)
                if captured:
                    name = PIECE_NAME.get(captured.symbol().lower(), "piece")
                    sq = chess.square_name(move.to_square)
                    capture_text = f"It captures the {name} on {sq}."
                else:
                    sq = chess.square_name(move.to_square)
                    capture_text = f"It captures on {sq}."
        tmp.push(move)
        check_text = ""
        if tmp.is_checkmate():
            check_text = "It ends the game immediately with checkmate."
        elif tmp.is_check():
            check_text = "It gives check."

        promo_text = ""
        if move.promotion:
            promo_piece = chess.piece_symbol(move.promotion).lower()
            promo_name = PIECE_NAME.get(promo_piece, "piece")
            promo_text = f"It promotes to a {promo_name}."

        castle_text = ""
        if san.startswith("O-O-O"):
            castle_text = "It castles long to bring the king to safety and connect the rooks."
        elif san.startswith("O-O"):
            castle_text = "It castles to bring the king to safety and connect the rooks."

        parts = [p for p in [castle_text, capture_text, promo_text, check_text] if p]
        return " ".join(parts) if parts else None
    
    def describe_cp_mate(cp, mate, turn_color):
        opp = "Black" if turn_color.lower() == "white" else "White"

        if mate is not None and mate != 0:
            if mate > 0:
                return f"This line leads to a forced checkmate in {mate} for {turn_color}."
            else:
                return f"This line allows {opp} to force checkmate in {abs(mate)}."

        a = abs(cp)
        if a < 20:
            return "Overall, the position stays about even."
        elif a < 80:
            desc = "a slight edge"
        elif a < 200:
            desc = "a clear advantage"
        elif a < 500:
            desc = "a big advantage"
        else:
            desc = "a winning advantage"

        pawns = cp / 100.0

        if cp > 0:
            return f"Overall, {turn_color} comes out with {desc} (about +{pawns:.2f} pawns)."
        else:
            return f"Overall, {turn_color} is worse here (about {pawns:.2f} pawns)."

    lines = []
    if shuffle_candidates:
        random.shuffle(candidates)
    for i, cand in enumerate(candidates):
        san = cand['san']
        pv = cand['pv']
        wr = cand['win_rate']
        cp = cand['cp']
        mate = cand['mate']

        
        if i == 0:
            lines.append(f"First, consider {san}.")
            lines.append(f"If played here, a natural continuation might be:")
        else:
            lines.append(f"Another possible choice is {san}.")
            lines.append(f"If played this way, the possible continuation is:")
            
        lines.append(pv)
        move_description = describe_candidate_move(san, board)
        if move_description:
            lines.append(move_description)
        lines.append(describe_cp_mate(cp, mate, turn_color))
        lines.append(f"In this variation, {turn_color}'s win rate is about {wr}%.")
        lines.append("")
    
    return "\n".join(lines).strip()

def format_reasoning_trace(info, template_str, shuffle_candidates=False):
    candidate_analysis_text = generate_candidate_text(info["fen"], info['candidates'], info['turn_color'], shuffle_candidates=shuffle_candidates)
    
    try:
        return template_str.format(
            fen=info['fen'],
            turn_color=info['turn_color'],
            candidate_analysis=candidate_analysis_text,
            best_move=info['best_move'],
            best_pv=info['best_pv'],
            best_win_rate=info['best_win_rate']
        )
    except KeyError as e:
        return f"Error: Template missing key {e}"


def format_pv_from_uci(board, uci_line, max_moves):
    temp_board = board.copy()
    uci_moves = uci_line.split(" ")
    
    lines = []
    for uci in uci_moves[:max_moves]: 
        move = chess.Move.from_uci(uci)
        san = temp_board.san(move)
        
        if temp_board.turn == chess.WHITE:
            lines.append(f"{san}")
        else:
            lines.append(f"... {san}")
        
        temp_board.push(move)

    return "\n".join(lines)

def parse_lichess_entry(json_line, max_candidates, max_moves):

    data = json.loads(json_line)

    evals_list = data['evals']
    
    # use the deepest eval for best move
    deepest_eval = max(evals_list, key=lambda x: x.get('depth', 0))
    best_move_truth = deepest_eval['pvs'][0]['line'].split()[0]
    
    # use the richest eval for multi-pv candidates
    richest_eval = max(evals_list, key=lambda x: (len(x['pvs']), x.get('depth', 0)))
    
    # Get the best move from the multi-pv search
    best_move_reasoning = richest_eval['pvs'][0]['line'].split()[0]
    
    if best_move_truth != best_move_reasoning:
        return None
        
    if len(richest_eval['pvs']) < 2:
        return None
    
    if richest_eval['pvs'][0].get('cp') is not None and richest_eval['pvs'][0].get('cp') == richest_eval['pvs'][1].get('cp'):
        return None
        
    fen = data['fen']
    board = chess.Board(fen)
    perspective = 1 if board.turn == chess.WHITE else -1
    
    candidates = []
    best_move_san = None
    best_win_rate = 0.0
    best_pv = ""

    for i, pv in enumerate(richest_eval['pvs']):
        if i >= max_candidates:
            break
        line_str = pv['line']
        formatted_pv = format_pv_from_uci(board, line_str, max_moves)
        uci_moves = line_str.split(" ")
        root_move_uci = uci_moves[0]
        
        move_obj = chess.Move.from_uci(root_move_uci)
        san_move = board.san(move_obj)

        raw_cp = pv.get('cp')     
        raw_mate = pv.get('mate')

        cp = raw_cp * perspective if raw_cp is not None else None
        mate = raw_mate * perspective if raw_mate is not None else None
        win_rate = get_win_rate(cp, mate)
        
        candidates.append({
            "san": san_move,
            "uci": root_move_uci,
            "win_rate": win_rate,
            "cp": cp,
            "mate": mate,
            "pv": formatted_pv
        })

        if i == 0:
            best_move_san = san_move
            best_win_rate = win_rate
            best_pv = formatted_pv

    return {
        "fen": fen,
        "turn_color": "White" if board.turn == chess.WHITE else "Black",
        "candidates": candidates,
        "best_move": best_move_san,
        "best_win_rate": best_win_rate,
        "best_pv": best_pv
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="./data/lichess_db_eval_head20k.jsonl")
    parser.add_argument("--max_candidates", type=int, default=3)
    parser.add_argument("--max_moves", type=int, default=5)
    parser.add_argument("--template_path", default="./reasoning_template.txt")
    parser.add_argument("--out_path", default="./data/lichess_eval_sft.jsonl")
    parser.add_argument("--shuffle_candidates", action='store_true')
    args = parser.parse_args()
    
    
    valid_count = 0
    skipped_count = 0
    
    with open(args.data_path, 'r') as f_in, open(args.out_path, 'w') as f_out:
        for line in f_in:
            if not line.strip(): continue
            
            stockfish_info = parse_lichess_entry(line, args.max_candidates, args.max_moves)
            
            if stockfish_info:
                output_text = format_reasoning_trace(stockfish_info, template_str=open(args.template_path).read(), shuffle_candidates=args.shuffle_candidates)
                


                sft_entry = {
                    "input": f"{stockfish_info['fen']}",
                    "best_move": stockfish_info['best_move'],
                    "output": output_text
                }
                
                f_out.write(json.dumps(sft_entry) + "\n")
                valid_count += 1
            else:
                skipped_count += 1

    print(f"Done.")
    print(f"Valid Samples Kept: {valid_count}")
    print(f"Samples Filtered Out: {skipped_count}")

if __name__ == "__main__":
    main()