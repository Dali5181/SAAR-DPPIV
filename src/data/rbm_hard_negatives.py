"""
RBM-based hard negative sample generation.
Trains a Restricted Boltzmann Machine on positive peptide samples to generate
adversarial negatives with similar physicochemical properties but no known activity.
"""
import numpy as np
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
MAX_SEQ_LEN = 30


def encode_sequence(seq: str, max_len: int = MAX_SEQ_LEN) -> np.ndarray:
    """One-hot encode a peptide sequence to a fixed-length binary vector."""
    vec = np.zeros(max_len * len(AMINO_ACIDS), dtype=np.float32)
    for pos, aa in enumerate(seq[:max_len]):
        if aa in AA_TO_IDX:
            vec[pos * len(AMINO_ACIDS) + AA_TO_IDX[aa]] = 1.0
    return vec


def decode_vector(vec: np.ndarray, max_len: int = MAX_SEQ_LEN) -> str:
    """Decode a continuous vector back to a peptide sequence."""
    seq = []
    n_aa = len(AMINO_ACIDS)
    for pos in range(max_len):
        start = pos * n_aa
        segment = vec[start:start + n_aa]
        if segment.max() < 0.1:
            break
        idx = segment.argmax()
        seq.append(AMINO_ACIDS[idx])
    return "".join(seq)


def compute_physicochemical(seq: str) -> dict:
    """Compute basic physicochemical properties for filtering."""
    hydrophobicity = {
        'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
        'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
        'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
        'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3,
    }
    mw = {
        'A': 89, 'C': 121, 'D': 133, 'E': 147, 'F': 165,
        'G': 75, 'H': 155, 'I': 131, 'K': 146, 'L': 131,
        'M': 149, 'N': 132, 'P': 115, 'Q': 146, 'R': 174,
        'S': 105, 'T': 119, 'V': 117, 'W': 204, 'Y': 181,
    }

    gravy = np.mean([hydrophobicity.get(aa, 0) for aa in seq]) if seq else 0
    molecular_weight = sum(mw.get(aa, 110) for aa in seq) - 18 * (len(seq) - 1)
    charge = sum(1 for aa in seq if aa in 'KR') - sum(1 for aa in seq if aa in 'DE')

    return {"gravy": gravy, "mw": molecular_weight, "charge": charge, "length": len(seq)}


class RBMNegativeGenerator:
    """Generate hard negative peptide samples using RBM."""

    def __init__(self, n_components=128, n_iter=50, learning_rate=0.01,
                 random_state=42):
        self.rbm = BernoulliRBM(
            n_components=n_components,
            n_iter=n_iter,
            learning_rate=learning_rate,
            random_state=random_state,
            verbose=0,
        )
        self.random_state = random_state
        self.positive_props = None

    def fit(self, positive_sequences: list):
        """Train RBM on positive peptide sequences."""
        print(f"  Training RBM on {len(positive_sequences)} positive sequences...")
        X = np.array([encode_sequence(seq) for seq in positive_sequences])
        self.rbm.fit(X)

        self.positive_props = {
            "gravy": [], "mw": [], "charge": [], "length": []
        }
        for seq in positive_sequences:
            props = compute_physicochemical(seq)
            for k in self.positive_props:
                self.positive_props[k].append(props[k])
        for k in self.positive_props:
            arr = np.array(self.positive_props[k])
            self.positive_props[k] = {"mean": arr.mean(), "std": arr.std(),
                                       "min": arr.min(), "max": arr.max()}

        print(f"  RBM training complete. Positive property ranges:")
        for k, v in self.positive_props.items():
            print(f"    {k}: {v['mean']:.2f} +/- {v['std']:.2f}")

    def generate(self, n_samples: int = 300, n_gibbs_steps: int = 5,
                 existing_positives: set = None, existing_negatives: set = None) -> list:
        """Generate hard negative samples via Gibbs sampling with mutation."""
        if existing_positives is None:
            existing_positives = set()
        if existing_negatives is None:
            existing_negatives = set()

        rng = np.random.RandomState(self.random_state)
        generated = set()
        all_known = existing_positives | existing_negatives

        positive_list = list(existing_positives)
        if not positive_list:
            print("  No positive sequences to base generation on.")
            return []

        print(f"  Generating {n_samples} hard negative samples...")

        # Strategy 1: RBM-guided perturbation of positive sequences
        for attempt in range(n_samples * 30):
            if len(generated) >= n_samples:
                break

            base_seq = positive_list[rng.randint(len(positive_list))]
            seq_list = list(base_seq)

            n_mutations = max(1, rng.poisson(max(1, len(seq_list) // 3)))
            n_mutations = min(n_mutations, len(seq_list))

            positions = rng.choice(len(seq_list), size=n_mutations, replace=False)
            for pos in positions:
                base_vec = encode_sequence("".join(seq_list))
                h = self.rbm._mean_hiddens(base_vec.reshape(1, -1))
                v_probs = self._visible_probs(h)

                n_aa = len(AMINO_ACIDS)
                start = pos * n_aa
                if start + n_aa <= len(v_probs[0]):
                    segment = v_probs[0, start:start + n_aa]
                    segment = np.clip(segment, 0, None)
                    noise = rng.uniform(0.1, 0.5, n_aa)
                    segment = segment + noise
                    segment /= segment.sum()
                    chosen = rng.choice(n_aa, p=segment)
                    seq_list[pos] = AMINO_ACIDS[chosen]

            op = rng.random()
            if op < 0.15 and len(seq_list) > 2:
                del_pos = rng.randint(len(seq_list))
                seq_list.pop(del_pos)
            elif op < 0.3 and len(seq_list) < MAX_SEQ_LEN:
                ins_pos = rng.randint(len(seq_list) + 1)
                ins_aa = AMINO_ACIDS[rng.randint(len(AMINO_ACIDS))]
                seq_list.insert(ins_pos, ins_aa)

            new_seq = "".join(seq_list)

            if len(new_seq) < 2 or len(new_seq) > MAX_SEQ_LEN:
                continue
            if new_seq in all_known or new_seq in generated:
                continue
            if not all(aa in AMINO_ACIDS for aa in new_seq):
                continue

            generated.add(new_seq)

        result = list(generated)
        print(f"  Generated {len(result)} hard negatives")
        return result

    def _visible_probs(self, h):
        """Compute visible unit probabilities from hidden states."""
        return 1.0 / (1.0 + np.exp(-(h @ self.rbm.components_ + self.rbm.intercept_visible_)))
