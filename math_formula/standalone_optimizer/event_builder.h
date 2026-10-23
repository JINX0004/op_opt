#pragma once

#include "optimizer.h"

namespace mmopt {

// Expand a finalized, caller-vetted plan into shape/tiling-derived work.
// Initial scope: contiguous ND, NN, no bias, FP16/BF16, BASE or
// DETERMINISTIC_SPLIT_K. The returned candidate contains generated events.
// This is a logical-work model; it does not prove actual kernel legality.
Candidate BuildMatmulEvents(const Workload& workload, Candidate plan);

}  // namespace mmopt
