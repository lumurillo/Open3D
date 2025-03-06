// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <iostream>
#include <filesystem>

#include <vector>

#include "open3d/core/Dtype.h"
#include "open3d/core/Tensor.h"
#include "open3d/io/FileFormatIO.h"
#include "open3d/t/geometry/TensorMap.h"
#include "open3d/t/io/PointCloudIO.h"
#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Logging.h"
#include "open3d/utility/ProgressReporters.h"

namespace open3d {
namespace t {
namespace io {

#define SH_C0 0.28209479177387814

struct AttributePtr {
    AttributePtr(const core::Dtype &dtype,
                 const void *data_ptr,
                 const int &group_size)
        : dtype_(dtype), data_ptr_(data_ptr), group_size_(group_size) {}

    const core::Dtype dtype_;
    const void *data_ptr_;
    const int group_size_;
};

// Sigmoid function for opacity calculation
inline double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

// Function to convert int to byte representation
unsigned char int_to_byte(const int val) {
    unsigned char byte_val = 0;

    if (val > std::numeric_limits<unsigned char>::min() && 
        val < std::numeric_limits<unsigned char>::max()) {
        byte_val = static_cast<unsigned char>(val); 
    }

    return byte_val;
}

template <typename scalar_t>
Eigen::Vector4i ComputeColor(const scalar_t *f_dc_ptr, const scalar_t *opacity_ptr) {
    Eigen::Vector4f color;

    color[0] = 0.5 + SH_C0 * f_dc_ptr[0];
    color[1] = 0.5 + SH_C0 * f_dc_ptr[1];
    color[2] = 0.5 + SH_C0 * f_dc_ptr[2];
    color[3] = sigmoid(*opacity_ptr);

    // Convert color to uint8 (scale, clip, and cast)
    return (color * 255.0).cwiseMin(255.0).cwiseMax(0.0).cast<int>();
}

bool splat_write_byte(FILE *splat_file, unsigned char val) {
    if (fprintf(splat_file, "%d", val) <= 0) {
        utility::LogWarning("Write SPLAT failed: Error writing to file");
        return false;
    }
    return true;
}

bool splat_write_float(FILE *splat_file, float val) {
    // Create a byte array to hold the 4 bytes
    unsigned char bytes[4];

    // Copy the float data into the byte array
    std::memcpy(bytes, &val, sizeof(val));

    for (int idx = 0; idx < 4; ++idx) {
        if (!splat_write_byte(splat_file, bytes[idx])) {
            return false;
        }
    }
    return true;
}

bool ValidSPLATData(const geometry::PointCloud &pointcloud,
                    geometry::TensorMap t_map) {

    // Positions
    if (t_map["positions"].GetDtype() != core::Float32) {
        utility::LogWarning("Write SPLAT failed: "
                            "unsupported data type: {}.",
                            t_map["positions"].GetDtype().ToString());
        return false;
    }

    // Scale
    if (pointcloud.HasPointScales()) {
        if (t_map["scale"].GetDtype() != core::Float32) {
            utility::LogWarning("Write SPLAT failed: "
                                "unsupported data type: {}.",
                                t_map["scale"].GetDtype().ToString());
            return false;
        }
    } else {
        utility::LogWarning("Write SPLAT failed: "
                            "couldn't find valid \"scale\" attribute.");
        return false;
    }

    // Rot
    if (!pointcloud.HasPointRots()) {
        utility::LogWarning("Write SPLAT failed: "
                            "couldn't find valid \"rot\" attribute.");
        return false;
    }

    // f_dc
    if (!pointcloud.HasPointFDCs()) {
        utility::LogWarning("Write SPLAT failed: "
                            "couldn't find valid \"f_dc\" attribute.");
        return false;
    }

    // Opacity
    if (!pointcloud.HasPointOpacities()) {
        utility::LogWarning("Write SPLAT failed: "
                            "couldn't find valid \"opacity\" attribute.");
        return false;
    }

    return true;
}

bool WritePointCloudToSPLAT(const std::string &filename,
                            const geometry::PointCloud &pointcloud,
                            const open3d::io::WritePointCloudOption &params) {
    FILE *splat_file = NULL;

    // Validate Point Cloud
    if (pointcloud.IsEmpty()) {
        utility::LogWarning("Write SPLAT failed: point cloud has 0 points.");
        return false;
    }

    // Validate Splat Data
    geometry::TensorMap t_map(pointcloud.GetPointAttr().Contiguous());
    if(!ValidSPLATData(pointcloud, t_map)) return false;

    // Take pointers to attribute data
    AttributePtr positions_attr = AttributePtr(t_map["positions"].GetDtype(),
                                               t_map["positions"].GetDataPtr(),
                                               3);
    AttributePtr scale_attr = AttributePtr(t_map["scale"].GetDtype(),
                                           t_map["scale"].GetDataPtr(), 3);

    AttributePtr rot_attr = AttributePtr(t_map["rot"].GetDtype(),
                                         t_map["rot"].GetDataPtr(), 4);

    AttributePtr f_dc_attr = AttributePtr(t_map["f_dc"].GetDtype(),
                                 t_map["f_dc"].GetDataPtr(), 3);
    
    AttributePtr opacity_attr = AttributePtr(t_map["opacity"].GetDtype(),
                                             t_map["opacity"].GetDataPtr(), 1);

    // Total Gaussians
    long num_gaussians =
            static_cast<long>(pointcloud.GetPointPositions().GetLength());

    // Open splat file
    splat_file = fopen(filename.c_str(), "wb");
    if (!splat_file) {  
        utility::LogWarning("Write SPLAT failed: unable to open file: {}.",
                            filename);
        return false;
    }

    // Write to SPLAT
    utility::CountingProgressReporter reporter(params.update_progress);
    reporter.SetTotal(num_gaussians);

    for (int64_t i = 0; i < num_gaussians; i++) {
        
        // Positions
        DISPATCH_DTYPE_TO_TEMPLATE(positions_attr.dtype_, [&]() {
            const scalar_t *positions_ptr =
                    static_cast<const scalar_t *>(positions_attr.data_ptr_);
            for (int idx_offset = positions_attr.group_size_ * i;
                idx_offset < positions_attr.group_size_ * (i + 1); ++idx_offset) {
                splat_write_float(splat_file, positions_ptr[idx_offset]);
            }
        });

        // Scale
        DISPATCH_DTYPE_TO_TEMPLATE(scale_attr.dtype_, [&]() {
            const scalar_t *scale_ptr =
                    static_cast<const scalar_t *>(scale_attr.data_ptr_);
            for (int idx_offset = scale_attr.group_size_ * i;
                idx_offset < scale_attr.group_size_ * (i + 1); ++idx_offset) {
                splat_write_float(splat_file, scale_ptr[idx_offset]);
            }
        });

        // Color
        DISPATCH_DTYPE_TO_TEMPLATE(opacity_attr.dtype_, [&]() {
            int f_dc_offset = f_dc_attr.group_size_ * i;
            int opacity_offset = opacity_attr.group_size_ * i;
            const scalar_t *f_dc_ptr =
                    static_cast<const scalar_t *>(f_dc_attr.data_ptr_);
            const scalar_t *opacity_ptr =
                    static_cast<const scalar_t *>(opacity_attr.data_ptr_);
            Eigen::Vector4i color = ComputeColor(f_dc_ptr + f_dc_offset,
                                                 opacity_ptr + opacity_offset);
            
            for (int idx = 0; idx < 4; ++idx) {
                splat_write_byte(splat_file, int_to_byte(color[idx]));
            }
        });

        // Rot
        DISPATCH_DTYPE_TO_TEMPLATE(rot_attr.dtype_, [&]() {
            Eigen::Vector4f rot;
            const scalar_t *rot_ptr =
                    static_cast<const scalar_t *>(rot_attr.data_ptr_);
            int rot_offset = rot_attr.group_size_ * i;

            rot << rot_ptr[rot_offset], rot_ptr[rot_offset + 1],
                   rot_ptr[rot_offset + 2], rot_ptr[rot_offset + 3];

            rot = (((rot / rot.norm()) * 128.0) + Eigen::Vector4f::Constant(128.0));
            Eigen::Vector4i int_rot = rot.cwiseMin(255.0).cwiseMax(0.0).cast<int>();

            for (int idx = 0; idx < 4; ++idx) {
                splat_write_byte(splat_file, int_to_byte(int_rot[idx]));
            }
        });

        if (i % 1000 == 0) {
            reporter.Update(i);
        }
    }

    // Close file
    reporter.Finish();
    fclose(splat_file);
    return true;
}

}  // namespace io
}  // namespace t
}  // namespace open3d
