/**
 * @file euclidean_distance.h
 * @brief Semantics-Constrained Pairwise Euclidean distance geometric invariant
 * @author Doncey Albin <plusk@mit.edu>
 * @date 15 May 2021
 */

#pragma once

#include "clipper/invariants/abstract.h"

namespace clipper {
namespace invariants {

  /**
   * @brief      Specialization of PairwiseInvariant to Euclidean distance in
   *             the real numbers using the 2-norm as the invariant.
   */
  class SemanticsConstrainedEuclideanDistance : public PairwiseInvariant
  {
  public:
    struct Params
    {
      double sigma = 0.01; ///< spread / "variance" of exponential kernel
      double epsilon = 0.06; ///< bound on consistency score, determines if inlier/outlier
      double mindist = 0; ///< minimum allowable distance between inlier points in the same dataset
    };
  public:
    SemanticsConstrainedEuclideanDistance(const Params& params)
    : params_(params) {}
    ~SemanticsConstrainedEuclideanDistance() = default;

    /**
     * @brief      Functor for pairwise invariant scoring function
     *
     * @param[in]  ai    Element i from dataset 1
     * @param[in]  aj    Element j from dataset 1
     * @param[in]  bi    Element i from dataset 2
     * @param[in]  bj    Element j from dataset 2
     *
     * @return     The consistency score for the association of (ai,bi) and (aj,bj)
     */
    double operator()(const Datum& ai, const Datum& aj, const Datum& bi, const Datum& bj) override;

  private:
    Params params_;
  };

  using SemanticsConstrainedEuclideanDistancePtr = std::shared_ptr<SemanticsConstrainedEuclideanDistance>;

} // ns invariants
} // ns clipper