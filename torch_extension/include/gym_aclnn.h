/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once
#include <ATen/ATen.h>
#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <c10/core/DeviceGuard.h>
#include <cstdlib>
#include <dlfcn.h>
#include <functional>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <utility>
#include <vector>

namespace gym_torch {
// Keep library handles resident: cached function pointers and in-flight kernels
// must remain valid. Search every vendor path, not a colon-joined pathname.
inline const std::vector<void *> &Libraries() {
  static const auto handles = [] {
    std::vector<std::string> names;
    for (const auto &env :
         {std::make_pair("ASCEND_CUSTOM_OPP_PATH", "/op_api/lib/"),
          std::make_pair("LD_LIBRARY_PATH", "/")}) {
      const char *value = std::getenv(env.first);
      if (!value)
        continue;
      std::istringstream paths(value);
      for (std::string path; std::getline(paths, path, ':');) {
        if (!path.empty())
          names.push_back(path + env.second + "libcust_opapi.so");
      }
    }
    names.insert(names.end(),
                 {"libcust_opapi.so", "libopapi.so", "libnnopbase.so"});
    std::vector<void *> result;
    for (const auto &name : names) {
      if (void *handle = dlopen(name.c_str(), RTLD_NOW | RTLD_LOCAL))
        result.push_back(handle);
    }
    return result;
  }();
  return handles;
}

template <typename T> T Symbol(const char *name) {
  for (void *library : Libraries()) {
    if (void *symbol = dlsym(library, name))
      return reinterpret_cast<T>(symbol);
  }
  TORCH_CHECK(false, "Cannot resolve ", name,
              "; install the matching PyPTO-Gym .run in the active CANN "
              "environment (custom locations require ASCEND_CUSTOM_OPP_PATH)");
}

// ACL_FORMAT_ND describes existing storage; it does not convert its layout.
// Query host metadata before enqueueing: internal formats such as FRACTAL_NZ
// can be logically contiguous while their physical elements are reordered.
inline void CheckNDStorage(const at::Tensor &tensor, const char *name) {
  TORCH_CHECK(tensor.device().is_privateuseone(), name, " must be an NPU tensor");
  const auto format = at_npu::native::get_npu_format(tensor);
  TORCH_CHECK(format == ACL_FORMAT_ND, name,
              " must use ND storage format; got NPU format ", format);
}

// Plain-storage check for operators whose public inputs may be 4-D: torch_npu
// tags host-copied 4-D tensors as NCHW by default even though the physical
// layout is contiguous row-major, so a strict ACL_FORMAT_ND comparison would
// reject legitimate inputs. Private formats (FRACTAL_*, NC1HWC0, ...) are still
// rejected; the caller-side contiguity/stride checks keep the ACL_FORMAT_ND
// descriptor built from sizes/strides truthful.
inline void CheckPlainStorage(const at::Tensor &tensor, const char *name) {
  TORCH_CHECK(tensor.device().is_privateuseone(), name, " must be an NPU tensor");
  const auto format = at_npu::native::get_npu_format(tensor);
  TORCH_CHECK(format == ACL_FORMAT_ND || format == ACL_FORMAT_NCHW ||
                  format == ACL_FORMAT_NHWC,
              name, " must use plain ND storage format; got NPU format ",
              format);
}

inline aclDataType DataType(at::ScalarType type) {
  switch (type) {
  case at::kBFloat16:
    return ACL_BF16;
  case at::kHalf:
    return ACL_FLOAT16;
  case at::kFloat:
    return ACL_FLOAT;
  case at::kInt:
    return ACL_INT32;
  case at::kLong:
    return ACL_INT64;
  case at::kByte:
    return ACL_UINT8;
  case at::kFloat8_e4m3fn:
    // ACL_FLOAT8_E4M3FN (acl/acl_base_rt.h); the enum name is not exported by
    // every CANN header set, so use the ABI value, matching the test bridges.
    return static_cast<aclDataType>(36);
  case at::kFloat8_e8m0fnu:
    // ACL_FLOAT8_E8M0, same reasoning as above.
    return static_cast<aclDataType>(37);
  default:
    TORCH_CHECK(false, "Unsupported ACL tensor dtype: ", type);
  }
}

struct DestroyTensor {
  using Function = aclnnStatus (*)(const aclTensor *);
  Function destroy;
  void operator()(aclTensor *tensor) const noexcept {
    if (tensor)
      destroy(tensor);
  }
};
using TensorHandle = std::unique_ptr<aclTensor, DestroyTensor>;

// Copy metadata on the caller thread. Keeping at::Tensor references in a host
// task can make their last release re-enter the same task queue from its
// worker.
struct TensorInfo {
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
  int64_t offset;
  int64_t storage_elements;
  void *data;
  aclDataType type;
  explicit TensorInfo(const at::Tensor &tensor)
      : sizes(tensor.sizes().vec()), strides(tensor.strides().vec()),
        offset(tensor.storage_offset()),
        storage_elements(tensor.storage().nbytes() / tensor.element_size()),
        data(const_cast<void *>(tensor.storage().data())),
        type(DataType(tensor.scalar_type())) {}
};

inline TensorHandle Tensor(const TensorInfo &tensor) {
  using Create =
      aclTensor *(*)(const int64_t *, uint64_t, aclDataType, const int64_t *,
                     int64_t, aclFormat, const int64_t *, uint64_t, void *);
  static const auto create = Symbol<Create>("aclCreateTensor");
  static const auto destroy =
      Symbol<DestroyTensor::Function>("aclDestroyTensor");
  auto *result = create(tensor.sizes.data(), tensor.sizes.size(), tensor.type,
                        tensor.strides.data(), tensor.offset, ACL_FORMAT_ND,
                        &tensor.storage_elements, 1, tensor.data);
  TORCH_CHECK(result, "aclCreateTensor failed");
  return TensorHandle(result, DestroyTensor{destroy});
}

using Execute = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
inline void ExecuteOp(Execute execute, uint64_t bytes, aclOpExecutor *executor,
                      c10_npu::NPUStream stream) {
  TORCH_CHECK(bytes <=
                  static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
              "ACLNN workspace is too large");
  const auto raw_stream = stream.stream(false);
  // Allocate directly on the launch stream. Constructing/freeing a Tensor on
  // the host worker can record an event on its default stream and re-enter the
  // same task queue, deadlocking a cold call.
  auto release = [](void *ptr) {
    c10_npu::NPUCachingAllocator::raw_delete(ptr);
  };
  std::unique_ptr<void, decltype(release)> workspace(
      bytes ? c10_npu::NPUCachingAllocator::raw_alloc_with_stream(bytes,
                                                                  raw_stream)
            : nullptr,
      release);
  const auto status = execute(workspace.get(), bytes, executor, raw_stream);
  TORCH_CHECK(status == 0, "ACLNN execution failed, status=", status, ": ",
              aclGetRecentErrMsg());
}

// Record stream use on the caller thread, just as tensor release/event
// recording is queued there. Worker tasks contain host metadata and
// stream-owned workspace only, matching torch_npu's asynchronous ACLNN dispatch
// contract.
using Task =
    std::function<int(const std::vector<TensorInfo> &, c10_npu::NPUStream)>;
inline void Enqueue(const char *name, const std::vector<at::Tensor> &tensors,
                    Task task) {
  const auto stream =
      c10_npu::getCurrentNPUStream(tensors.front().device().index());
  std::vector<TensorInfo> metadata;
  for (const auto &tensor : tensors) {
    if (tensor.numel())
      c10_npu::NPUCachingAllocator::recordStream(tensor.storage().data_ptr(),
                                                 stream);
    metadata.emplace_back(tensor);
  }
  const auto raw_stream = stream.stream(false);
  if (c10_npu::check_enqueue_need_use(raw_stream))
    TORCH_CHECK(aclrtUseStreamResInCurrentThread(raw_stream) == ACL_SUCCESS,
                "Cannot use NPU stream resources in the current thread");
  auto callback = [metadata = std::move(metadata), stream,
                   task = std::move(task)] {
    const c10::OptionalDeviceGuard guard(stream.device());
    const auto raw_stream = stream.stream(false);
    if (c10_npu::check_dequeue_need_use(raw_stream))
      TORCH_CHECK(aclrtUseStreamResInCurrentThread(raw_stream) == ACL_SUCCESS,
                  "Cannot use NPU stream resources in the current thread");
    return task(metadata, stream);
  };
  at_npu::native::OpCommand::RunOpApiV2(name, callback);
}
} // namespace gym_torch
