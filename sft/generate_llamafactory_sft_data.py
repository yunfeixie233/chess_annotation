import argparse
import json
import math
import chess
import random
import os

PIECE_NAME = {
    "p": "pawn",
    "n": "knight",
    "b": "bishop",
    "r": "rook",
    "q": "queen",
    "k": "king",
}


class PromptTemplates:
    """Manages instruction/prompt variants for diversity."""
    
    VARIANTS = {
        "v1": [
            "Analyze this chess position and determine the best move. Explain your reasoning by examining candidate moves and their consequences.",
            "Given this chess position, find the optimal move. Consider multiple candidate moves and explain which one is best.",
            "Evaluate this chess position and identify the strongest continuation. Analyze the key candidate moves before deciding.",
            "Study this chess position carefully. What is the best move? Explain your analysis of the candidate options.",
            "Look at this chess position and determine the best move for the side to play. Walk through your reasoning step by step.",
        ],
        "v2": [
            "What is the best move in this position? Analyze the candidates and explain your choice.",
            "Find the strongest move here. Compare the main options and justify your selection.",
            "Determine the optimal continuation. Evaluate the candidate moves and their outcomes.",
            "Identify the best move in this position. Provide analysis of the key alternatives.",
            "Calculate the best continuation. Examine the candidate moves and explain your reasoning.",
        ],
    }
    
    SYSTEM_PROMPTS = {
        "v1": "You are a chess analysis assistant. Given a chess position in FEN notation, analyze the position and find the best move.",
        "v2": "You are an expert chess engine. Analyze positions deeply and find the strongest moves.",
    }
    
    def __init__(self, prompt_class="v1"):
        self.prompt_class = prompt_class
        self.variants = self.VARIANTS[prompt_class]
        self.system_prompt = self.SYSTEM_PROMPTS[prompt_class]
    
    def get_random_instruction(self):
        return random.choice(self.variants)
    
    def get_system_prompt(self):
        return self.system_prompt


class ResponseTemplates:
    """Manages response/reasoning templates for diversity."""
    
    VARIANTS = {
        "v1": [
            """<think>
The current position is: {fen}.
The next player is {turn_color}.

Let's analyze {turn_color}'s possible next moves:
{candidate_analysis}

Based on the above analysis, the best move is {best_move}, with a {best_win_rate}% win rate.
Choosing this move, the possible continuation is:
{best_pv}
</think>
<answer>
\\boxed{{{best_move}}}
</answer>""",
            """<think>
Position (FEN): {fen}
To move: {turn_color}

Examining the candidate moves for {turn_color}:
{candidate_analysis}

After considering all options, {best_move} stands out as the strongest choice with a {best_win_rate}% win rate.
The main line continues:
{best_pv}
</think>
<answer>
\\boxed{{{best_move}}}
</answer>""",
            """<think>
Looking at this position: {fen}
It is {turn_color}'s turn to move.

What are the key options here?
{candidate_analysis}

The conclusion is clear: {best_move} is the optimal move, giving {turn_color} a {best_win_rate}% chance of winning.
Here is the expected continuation:
{best_pv}
</think>
<answer>
\\boxed{{{best_move}}}
</answer>""",
            """<think>
Given the position {fen}, {turn_color} is to play.

Let me evaluate the critical continuations:
{candidate_analysis}

My recommendation is {best_move}. This move yields a {best_win_rate}% win probability for {turn_color}.
The likely follow-up is:
{best_pv}
</think>
<answer>
\\boxed{{{best_move}}}
</answer>""",
            """<think>
FEN: {fen}
Side to move: {turn_color}

Analyzing the position, here are the main candidate moves:
{candidate_analysis}

Taking everything into account, the best continuation is {best_move}, offering a {best_win_rate}% win rate.
The game might proceed as follows:
{best_pv}
</think>
<answer>
\\boxed{{{best_move}}}
</answer>""",
        ],
        "v2": [
            """<think>
Position: {fen}
{turn_color} to move.

Candidate analysis:
{candidate_analysis}

Conclusion: {best_move} is best with {best_win_rate}% win rate.
Line: {best_pv}
</think>
<answer>
\\boxed{{{best_move}}}
</answer>""",
            """<think>
Analyzing {fen} for {turn_color}:

{candidate_analysis}

Best: {best_move} ({best_win_rate}% win rate)
Continuation: {best_pv}
</think>
<answer>
\\boxed{{{best_move}}}
</answer>""",
            """<think>
{turn_color} to play in: {fen}

Options:
{candidate_analysis}

Verdict: Play {best_move} for {best_win_rate}% winning chances.
Expected: {best_pv}
</think>
<answer>
\\boxed{{{best_move}}}
</answer>""",
            """<think>
{fen}
Turn: {turn_color}

{candidate_analysis}

Result: {best_move} wins with {best_win_rate}%.
Follow-up: {best_pv}
</think>
<answer>
\\boxed{{{best_move}}}
</answer>""",
            """<think>
Evaluating position for {turn_color}: {fen}

{candidate_analysis}

Decision: {best_move} is optimal ({best_win_rate}% win rate).
Main line: {best_pv}
</think>
<answer>
\\boxed{{{best_move}}}
</answer>""",
        ],
    }
    
    def __init__(self, template_class="v1"):
        self.template_class = template_class
        self.variants = self.VARIANTS[template_class]
    
    def get_random_template(self):
        return random.choice(self.variants)


class CandidateAnalysisGenerator:
    """Generates candidate move analysis text."""
    
    def __init__(self):
        pass
    
    def describe_candidate_move(self, san, board):
        tmp = board.copy()
        move = tmp.parse_san(san)
        capture_text = ""
        if tmp.is_capture(move):
            if tmp.is_en_passant(move):
                capture_text = "It captures a pawn en passant."
            else:
                captured = tmp.piece_at(move.to_square)
                if captured:
                    name = PIECE_NAME[captured.symbol().lower()]
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
            promo_name = PIECE_NAME[promo_piece]
            promo_text = f"It promotes to a {promo_name}."

        castle_text = ""
        if san.startswith("O-O-O"):
            castle_text = "It castles long to bring the king to safety and connect the rooks."
        elif san.startswith("O-O"):
            castle_text = "It castles to bring the king to safety and connect the rooks."

        parts = [p for p in [castle_text, capture_text, promo_text, check_text] if p]
        return " ".join(parts) if parts else None
    
    def describe_cp_mate(self, cp, mate, turn_color):
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

    def generate(self, fen, candidates, turn_color, shuffle_candidates=False):
        board = chess.Board(fen)
        
        lines = []
        if shuffle_candidates:
            candidates = candidates.copy()
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
            move_description = self.describe_candidate_move(san, board)
            if move_description:
                lines.append(move_description)
            lines.append(self.describe_cp_mate(cp, mate, turn_color))
            lines.append(f"In this variation, {turn_color}'s win rate is about {wr}%.")
            lines.append("")
        
        return "\n".join(lines).strip()


class LichessDataParser:
    """Parses Lichess evaluation data."""
    
    def __init__(self, max_candidates=3, max_moves=5):
        self.max_candidates = max_candidates
        self.max_moves = max_moves
    
    def get_win_rate(self, cp, mate):
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

    def format_pv_from_uci(self, board, uci_line):
        temp_board = board.copy()
        uci_moves = uci_line.split(" ")
        
        lines = []
        for uci in uci_moves[:self.max_moves]: 
            move = chess.Move.from_uci(uci)
            san = temp_board.san(move)
            
            if temp_board.turn == chess.WHITE:
                lines.append(f"{san}")
            else:
                lines.append(f"... {san}")
            
            temp_board.push(move)

        return "\n".join(lines)

    def parse(self, json_line):
        data = json.loads(json_line)
        evals_list = data['evals']
        
        deepest_eval = max(evals_list, key=lambda x: x.get('depth', 0))
        best_move_truth = deepest_eval['pvs'][0]['line'].split()[0]
        
        richest_eval = max(evals_list, key=lambda x: (len(x['pvs']), x.get('depth', 0)))
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
            if i >= self.max_candidates:
                break
            line_str = pv['line']
            formatted_pv = self.format_pv_from_uci(board, line_str)
            uci_moves = line_str.split(" ")
            root_move_uci = uci_moves[0]
            
            move_obj = chess.Move.from_uci(root_move_uci)
            san_move = board.san(move_obj)

            raw_cp = pv.get('cp')     
            raw_mate = pv.get('mate')

            cp = raw_cp * perspective if raw_cp is not None else None
            mate = raw_mate * perspective if raw_mate is not None else None
            win_rate = self.get_win_rate(cp, mate)
            
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


class SFTDataGenerator:
    """Main generator for SFT data in LlamaFactory format."""
    
    def __init__(self, prompt_class="v1", template_class="v1", 
                 max_candidates=3, max_moves=5, shuffle_candidates=False, include_system=False):
        self.prompt_templates = PromptTemplates(prompt_class)
        self.response_templates = ResponseTemplates(template_class)
        self.candidate_generator = CandidateAnalysisGenerator()
        self.parser = LichessDataParser(max_candidates, max_moves)
        self.shuffle_candidates = shuffle_candidates
        self.include_system = include_system
        
        self.prompt_class = prompt_class
        self.template_class = template_class
        
        self.stats = {
            "valid": 0,
            "skipped": 0,
            "prompt_usage": {},
            "template_usage": {},
        }
    
    def format_response(self, info):
        template_str = self.response_templates.get_random_template()
        
        # Track template usage
        template_idx = self.response_templates.variants.index(template_str)
        self.stats["template_usage"][template_idx] = self.stats["template_usage"].get(template_idx, 0) + 1
        
        candidate_analysis = self.candidate_generator.generate(
            info["fen"], info['candidates'], info['turn_color'], 
            shuffle_candidates=self.shuffle_candidates
        )
        
        return template_str.format(
            fen=info['fen'],
            turn_color=info['turn_color'],
            candidate_analysis=candidate_analysis,
            best_move=info['best_move'],
            best_pv=info['best_pv'],
            best_win_rate=info['best_win_rate']
        )
    
    def generate_entry(self, json_line):
        info = self.parser.parse(json_line)
        
        if info is None:
            self.stats["skipped"] += 1
            return None
        
        instruction = self.prompt_templates.get_random_instruction()
        
        # Track prompt usage
        prompt_idx = self.prompt_templates.variants.index(instruction)
        self.stats["prompt_usage"][prompt_idx] = self.stats["prompt_usage"].get(prompt_idx, 0) + 1
        
        output_text = self.format_response(info)
        
        sft_entry = {
            "instruction": instruction,
            "input": info['fen'],
            "output": output_text
        }
        
        if self.include_system:
            sft_entry["system"] = self.prompt_templates.get_system_prompt()
        
        self.stats["valid"] += 1
        return sft_entry
    
    def process_file(self, input_path, output_path):
        with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
            for line in f_in:
                if not line.strip():
                    continue
                
                entry = self.generate_entry(line)
                if entry:
                    f_out.write(json.dumps(entry) + "\n")
        
        return self.stats
    
    def get_output_filename(self, base_name="lichess_eval_llamafactory"):
        return f"{base_name}_pro{self.prompt_class}_res{self.template_class}.jsonl"


def main():
    parser = argparse.ArgumentParser(description="Generate SFT data for LlamaFactory from Lichess evaluations")
    parser.add_argument("--data_path", default="./data/lichess_db_eval_head20k.jsonl",
                        help="Path to input Lichess evaluation data")
    parser.add_argument("--out_dir", default="./data",
                        help="Output directory for generated data")
    parser.add_argument("--max_candidates", type=int, default=3,
                        help="Maximum number of candidate moves to analyze")
    parser.add_argument("--max_moves", type=int, default=5,
                        help="Maximum moves in PV line")
    parser.add_argument("--prompt_class", default="v1", choices=["v1", "v2"],
                        help="Prompt/instruction template class (each has 5 variants)")
    parser.add_argument("--template_class", default="v1", choices=["v1", "v2"],
                        help="Response template class (each has 5 variants)")
    parser.add_argument("--shuffle_candidates", action='store_true',
                        help="Randomly shuffle candidate move order")
    parser.add_argument("--include_system", action='store_true',
                        help="Include system prompt in output")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    generator = SFTDataGenerator(
        prompt_class=args.prompt_class,
        template_class=args.template_class,
        max_candidates=args.max_candidates,
        max_moves=args.max_moves,
        shuffle_candidates=args.shuffle_candidates,
        include_system=args.include_system
    )
    
    output_filename = generator.get_output_filename()
    output_path = os.path.join(args.out_dir, output_filename)
    
    print(f"Generating SFT data...")
    print(f"  Prompt class: {args.prompt_class} (5 variants)")
    print(f"  Template class: {args.template_class} (5 variants)")
    print(f"  Output: {output_path}")
    
    stats = generator.process_file(args.data_path, output_path)
    
    print(f"\nDone!")
    print(f"  Valid samples: {stats['valid']}")
    print(f"  Skipped samples: {stats['skipped']}")
    print(f"  Prompt usage: {stats['prompt_usage']}")
    print(f"  Template usage: {stats['template_usage']}")
    print(f"\nOutput saved to: {output_path}")


if __name__ == "__main__":
    main()
