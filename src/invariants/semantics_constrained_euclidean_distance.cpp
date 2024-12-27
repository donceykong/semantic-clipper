/**
 * @file semantics_constrained_euclidean_distance.cpp
 * @brief Pairwise semantic invariant between point classes (e.g., car point and car point)
 * @author Doncey Albin <donceyalbin@colorado.edu>
 * @date 26 August 2024
 */

#include "clipper/invariants/semantics_constrained_euclidean_distance.h"
#include <iostream> // Include this at the top of the file

namespace clipper {
namespace invariants {

double SemanticsConstrainedEuclideanDistance::operator()(const Datum& ai, const Datum& aj,
                                                         const Datum& bi, const Datum& bj)
{
  // Check if the size is at least 4 (to safely access the fourth element)
  if (ai.size() < 4 || aj.size() < 4 || bi.size() < 4 || bj.size() < 4) {
    std::cerr << "\033[1;31mError: One of the input vectors does not have a size of 4. All input elements need (x, y, z, label).\033[0m\n\n" << std::endl;
    return 0.0;
  }

  // Access the semantic label element of each vector
  const double ai_label = ai(3);
  const double aj_label = aj(3);
  const double bi_label = bi(3);
  const double bj_label = bj(3);

  // // Check if the fourth elements are all the same
  // if (!(ai_label == aj_label && ai_label == bi_label && ai_label == bj_label)) {
  //   return 0.0;
  // }

  // Verify that the semantic labels in corresponding set of points are the same
  if (!(ai_label == bi_label && aj_label == bj_label)) {
    return 0.0;
  }

  // Set variables to only their x, y, z elements
  Datum ai_xyz = ai.head(3);
  Datum aj_xyz = aj.head(3);
  Datum bi_xyz = bi.head(3);
  Datum bj_xyz = bj.head(3);

  // distance between two points in the same cloud
  const double l1 = (ai_xyz - aj_xyz).norm();
  const double l2 = (bi_xyz - bj_xyz).norm();

  // enforce minimum distance criterion -- if points in the same dataset
  // are too close, then this pair of associations cannot be selected
  if (params_.mindist > 0 && (l1 < params_.mindist || l2 < params_.mindist)) {
    return 0.0;
  }

  // consistency score
  const double c = std::abs(l1 - l2);

  return (c<params_.epsilon) ? std::exp(-0.5*c*c/(params_.sigma*params_.sigma)) : 0;
}

} // ns invariants
} // ns clipper

