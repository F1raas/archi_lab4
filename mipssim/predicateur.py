from typing import *
import enum


class BranchDecision(enum.Enum):
    TAKEN = 1
    NOT_TAKEN = 0
    UNDEFINED = None

    def __int__(self):
        if self == BranchDecision.TAKEN:
            return 1
        elif self == BranchDecision.NOT_TAKEN:
            return 0
        raise ValueError("Decision is not recognized")

    def __bool__(self) -> bool:
        return self == BranchDecision.TAKEN

    @staticmethod
    def from_bool(branch_taken: bool) -> "BranchDecision":
        return BranchDecision.TAKEN if branch_taken else BranchDecision.NOT_TAKEN


class BranchState(enum.Enum):
    STRONGLY_TAKEN = 0b11
    WEAKLY_TAKEN = 0b10
    WEAKLY_NOT_TAKEN = 0b01
    STRONGLY_NOT_TAKEN = 0b00

    def __int__(self) -> int:
        return self.value

    def predict(self) -> BranchDecision:
        return BranchDecision.TAKEN if self.value >= 2 else BranchDecision.NOT_TAKEN

    def update(self, branch_taken: bool) -> "BranchState":
        if branch_taken and self.value < 3:
            return BranchState(self.value + 1)
        elif not branch_taken and self.value > 0:
            return BranchState(self.value - 1)
        return self


class BranchPredictor:
    def __init__(self, initial_state: BranchState, num_entries: int):
        if num_entries <= 0:
            raise ValueError("Branch predictor must have at least one entry.")
        self.num_entries = num_entries
        self.states = [initial_state] * num_entries
        self.prediction_count = 0

    def get_prediction_count(self) -> int:
        return self.prediction_count

    def set_prediction_count(self, count: int):
        self.prediction_count = count

    def increment_prediction_count(self):
        self.prediction_count += 1

    def predict(self, program_counter: int) -> BranchDecision:
        index = program_counter % self.num_entries
        return self.states[index].predict()

    def update(self, program_counter: int, decision: BranchDecision):
        index = program_counter % self.num_entries
        self.states[index] = self.states[index].update(decision == BranchDecision.TAKEN)

    def get_states(self) -> List[BranchState]:
        return self.states

    def __str__(self):
        return "\n".join(f"{i}: {state}" for i, state in enumerate(self.states))


class HybridBranchPredictor:
    def __init__(self, num_tables: int, entries_per_table: int, history_length: int,
                 tag_bit_length: int, initial_state: BranchState, base_predictor_entries: int):
        if entries_per_table > (2 ** tag_bit_length):
            raise ValueError("Entries per table exceed the tag range.")

        self.base_predictor = BranchPredictor(initial_state, base_predictor_entries)
        self.history_register = [0] * history_length
        self.tag_bit_length = tag_bit_length
        self.tables = [
            {tag: BranchPredictor(initial_state, 1) for tag in range(entries_per_table)}
            for _ in range(num_tables)
        ]
        self.usage_counters = [0] * (num_tables + 1)

    def get_tables(self) -> List[Dict[int, BranchPredictor]]:
        return self.tables

    def get_history_register(self) -> List[int]:
        return self.history_register

    def get_base_predictor(self) -> BranchPredictor:
        return self.base_predictor

    def compute_tag(self, program_counter: int, history: List[int]) -> int:
        return sum((i + 1) * program_counter * bit for i, bit in enumerate(history)) % (2 ** self.tag_bit_length)

    def __str__(self) -> str:
        tables_info = "\n".join(
            f"Table {i + 1}:\n" +
            "\n".join(f"  Tag {tag}: {predictor.get_states()[0]}" for tag, predictor in table.items())
            for i, table in enumerate(self.tables)
        )
        return f"Base Predictor:\n{self.base_predictor}\nHistory Register: {self.history_register}\n{tables_info}"

    def predict(self, program_counter: int) -> Tuple[BranchDecision, Dict[str, Any]]:
        for table_index, table in enumerate(reversed(self.tables), start=1):
            tag = self.compute_tag(program_counter, self.history_register[-table_index:])
            if tag in table:
                prediction = table[tag].predict(program_counter)
                self.usage_counters[len(self.tables) - table_index + 1] += 1
                return prediction, {"table": len(self.tables) - table_index, "tag": tag, "pc": program_counter}

        prediction = self.base_predictor.predict(program_counter)
        self.usage_counters[0] += 1
        return prediction, {"table": 0, "pc": program_counter}

    def update(self, update_info: Dict[str, Any], branch_taken: BranchDecision):
        self.history_register.pop(0)
        self.history_register.append(int(branch_taken))

        table = update_info["table"]
        tag = update_info.get("tag")
        program_counter = update_info["pc"]

        if table == 0:
            self.base_predictor.update(program_counter, branch_taken)
        elif tag is not None:
            self.tables[table - 1][tag].update(program_counter, branch_taken)

def test_hybrid_branch_predictor():
    num_tables = 2
    entries_per_table = 4
    history_length = 2
    tag_bit_length = 2
    initial_state = BranchState.WEAKLY_TAKEN
    base_predictor_entries = 8

    predictor = HybridBranchPredictor(
        num_tables=num_tables,
        entries_per_table=entries_per_table,
        history_length=history_length,
        tag_bit_length=tag_bit_length,
        initial_state=initial_state,
        base_predictor_entries=base_predictor_entries
    )

    test_branches = [
        (100, True),
        (104, False),
        (100, True),
        (108, True),
        (112, False),
        (100, False),
        (104, True),
        (108, False),
        (112, True),
        (100, True),
    ]

    for pc, taken in test_branches:
        print("\n---")
        print(f"Branch at PC={pc} is {'TAKEN' if taken else 'NOT TAKEN'}")
        
        prediction, info = predictor.predict(pc)
        print(f"Prediction: {prediction.name}")

        actual_decision = BranchDecision.TAKEN if taken else BranchDecision.NOT_TAKEN
        predictor.update(info, actual_decision)

    print("\nFinal Predictor States:")
    print(predictor)

if __name__ == "__main__":
    test_hybrid_branch_predictor()
