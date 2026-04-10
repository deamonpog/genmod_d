"""Wolfram class metadata for all 256 elementary cellular automata rules.

Wolfram classes:
  I   - Homogeneous: evolve to uniform state
  II  - Periodic: evolve to simple periodic structures
  III - Chaotic: appear random/chaotic
  IV  - Complex: localized structures, long transients (edge of chaos)

Equivalence classes: rules related by left-right reflection and/or
bit complement produce equivalent dynamics (88 equivalence classes).
"""

from typing import Dict, FrozenSet, List, Set


def reflect_rule(rule: int) -> int:
    """Left-right reflection of an ECA rule.

    Swaps the role of left and right neighbors in the lookup table.
    """
    rule_bin = f"{rule:08b}"
    # Original ordering: 111,110,101,100,011,010,001,000
    # After reflecting neighborhoods (swap L and R):
    # 111->111, 110->011, 101->101, 100->001, 011->110, 010->010, 001->100, 000->000
    # So new bits at positions: 0->0, 1->4, 2->2, 3->6, 4->1, 5->5, 6->3, 7->7
    perm = [0, 4, 2, 6, 1, 5, 3, 7]
    new_bits = ["0"] * 8
    for i, j in enumerate(perm):
        new_bits[j] = rule_bin[i]
    return int("".join(new_bits), 2)


def complement_rule(rule: int) -> int:
    """Bit complement of an ECA rule.

    Swaps 0s and 1s in both inputs and outputs.
    """
    rule_bin = f"{rule:08b}"
    # Complement: flip all bits and reverse the order of entries
    new_bits = [str(1 - int(b)) for b in reversed(rule_bin)]
    return int("".join(new_bits), 2)


def get_equivalence_class(rule: int) -> FrozenSet[int]:
    """Return the set of equivalent rules under reflection + complement."""
    r = rule
    rc = complement_rule(r)
    rr = reflect_rule(r)
    rrc = complement_rule(rr)
    return frozenset({r, rc, rr, rrc})


def compute_all_equivalence_classes() -> List[FrozenSet[int]]:
    """Compute all 88 equivalence classes for the 256 ECA rules."""
    seen: Set[int] = set()
    classes: List[FrozenSet[int]] = []
    for rule in range(256):
        if rule not in seen:
            eq_class = get_equivalence_class(rule)
            classes.append(eq_class)
            seen.update(eq_class)
    return classes


# Wolfram classification for all 256 ECA rules.
# Sources: Wolfram's "A New Kind of Science" (2002), Table p.231;
# Li & Packard (1990); Wolfram MathWorld.
# Class assignments for ambiguous rules follow the majority convention.
_CLASS_I_RULES = {0, 8, 32, 40, 64, 96, 128, 136, 160, 168, 192, 224, 234, 235,
                  238, 239, 248, 249, 250, 251, 252, 253, 254, 255}

_CLASS_II_RULES = {1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 19, 23, 24,
                   25, 26, 27, 28, 29, 33, 34, 35, 36, 37, 38, 42, 43, 44, 46,
                   50, 51, 56, 57, 58, 62, 72, 73, 74, 76, 77, 78, 94, 104, 108,
                   130, 132, 134, 138, 140, 142, 152, 154, 156, 162, 164, 170,
                   172, 176, 178, 184, 200, 204, 232}

_CLASS_III_RULES = {18, 22, 30, 45, 60, 75, 86, 89, 90, 101, 102, 105, 106, 109,
                    120, 121, 122, 126, 129, 131, 133, 135, 137, 146, 149, 150,
                    151, 153, 161, 165, 169, 181, 182, 183, 195}

_CLASS_IV_RULES = {41, 54, 106, 110, 124}

# Some rules appear in published lists under different classes depending on
# initial conditions. We resolve conflicts by giving Class IV priority
# (rarer, more interesting), then III, then II, then I.
# Build the mapping, resolving duplicates in priority order.
_WOLFRAM_CLASS: Dict[int, int] = {}
for r in range(256):
    _WOLFRAM_CLASS[r] = 2  # default to Class II (most common)
for r in _CLASS_I_RULES:
    _WOLFRAM_CLASS[r] = 1
for r in _CLASS_II_RULES:
    _WOLFRAM_CLASS[r] = 2
for r in _CLASS_III_RULES:
    _WOLFRAM_CLASS[r] = 3
for r in _CLASS_IV_RULES:
    _WOLFRAM_CLASS[r] = 4


def get_wolfram_class(rule: int) -> int:
    """Return Wolfram class (1-4) for a given ECA rule number."""
    if not 0 <= rule <= 255:
        raise ValueError(f"Rule must be in [0, 255], got {rule}")
    return _WOLFRAM_CLASS[rule]


def get_wolfram_class_name(cls: int) -> str:
    """Human-readable name for a Wolfram class."""
    names = {1: "I (Homogeneous)", 2: "II (Periodic)", 3: "III (Chaotic)", 4: "IV (Complex)"}
    return names[cls]


def rules_by_wolfram_class() -> Dict[int, List[int]]:
    """Return dict mapping Wolfram class -> sorted list of rule numbers."""
    result: Dict[int, List[int]] = {1: [], 2: [], 3: [], 4: []}
    for rule in range(256):
        result[_WOLFRAM_CLASS[rule]].append(rule)
    return result


EQUIVALENCE_CLASSES = compute_all_equivalence_classes()
