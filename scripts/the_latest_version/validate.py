from typing import Tuple
import re

import numpy as np
from helper import Matrix, Tensor3D, get_matrix

# Fitness shaping in rank units. A loaded-path child must beat the rank it
# reuses; equal or worse rank always pays the full non-improvement penalty.
NO_IMPROVEMENT_PENALTY = 7.0


def validate(
    result: Tuple[np.ndarray, str, str]
) -> dict[str, float | str]:
    context = get_matrix()
    result, report, best_path = result
    full_report = report + best_path
    loaded_path_match = re.search(r"\bloaded_path_rank:\s*(\d+)", full_report)
    loaded_match = re.search(r"\bloaded_rank:\s*(\d+)", full_report)
    initial_match = re.search(r"\binitial_rank:\s*(\d+)", full_report)
    found_rank = int(result.shape[0])

    # Strategy descriptor for MAP-Elites: the rank at which the program started
    # searching. Ab-initio runs start at initial_rank (~1701); saved-path
    # refiners start at their branch point (varies by path+margin). Exposing it
    # as a behavior dimension keeps distinct load strategies in distinct archive
    # cells instead of collapsing them into one fitness/runtime niche.
    if loaded_match is not None:
        loaded_rank = float(loaded_match.group(1))
    elif initial_match is not None:
        loaded_rank = float(initial_match.group(1))
    else:
        loaded_rank = 1701.0

    # Retention must know whether this search actually beat the rank it was
    # asked to improve.  Saved-path runs use their loaded path's final rank;
    # ab-initio runs use their initial rank.  This is intentionally separate
    # from fitness so the archive selector can enforce it before fitness.
    rank_to_beat_match = loaded_path_match or loaded_match or initial_match
    rank_improved = float(
        rank_to_beat_match is not None
        and found_rank < int(rank_to_beat_match.group(1))
    )

    if np.any(Tensor3D(context) != Tensor3D(Matrix.from_numpy(result))):
        return {"fitness": 1290.0,
                "is_valid": 0.0,
                "loaded_rank": loaded_rank,
                "rank_improved": 0.0,
                "aux info": full_report,
                }

    base_fitness = result.shape[0] + np.sum(result) / np.size(result)
    penalty = 0.0
    # Keep GF16's existing fitness shaping unchanged: only a reused/loaded
    # branch pays the no-improvement penalty.  ``rank_improved`` above is
    # nevertheless available for archive retention on both fresh and loaded
    # searches.
    fitness_rank_to_beat_match = loaded_path_match or loaded_match
    if fitness_rank_to_beat_match is not None:
        rank_to_beat = int(fitness_rank_to_beat_match.group(1))
        if found_rank >= rank_to_beat:
            penalty += NO_IMPROVEMENT_PENALTY

    return {"fitness": base_fitness + penalty,
            "is_valid": 1.0,
            "loaded_rank": loaded_rank,
            "rank_improved": rank_improved,
            "aux info": full_report,
            }
