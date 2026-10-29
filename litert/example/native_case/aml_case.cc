/*
 * Copyright 2025 The Google AI Edge Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// Copyright (C) 2026 Amlogic, Inc.
// SPDX-License-Identifier: Apache-2.0

// JIT case: load a .tflite, fill inputs from a file or with random bytes, run
// once. Links libLiteRt.so. The compiler plugin and dispatch libraries are
// loaded at runtime from the current directory.

#ifndef NDEBUG
#define NDEBUG
#endif
#ifndef LITERT_NO_ABSL
#define LITERT_NO_ABSL
#endif

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "litert/c/internal/litert_runtime_api_export.h"
#include "litert/c/internal/litert_runtime_builtin.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_environment_options.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/options/litert_aml_options.h"

extern "C" const LiteRtRuntimeCApiStruct* GetLiteRtRuntimeBuiltin() {
  return &kLiteRtRuntimeBuiltin;
}

static void* ReadFile(const char* file_path, size_t* file_size) {
  FILE* fp = fopen(file_path, "rb");
  if (fp == nullptr) {
    std::cerr << "Failed to open input file: " << file_path << std::endl;
    return nullptr;
  }
  if (fseek(fp, 0, SEEK_END) != 0) {
    fclose(fp);
    std::cerr << "Failed to seek input file: " << file_path << std::endl;
    return nullptr;
  }
  const long size = ftell(fp);
  rewind(fp);
  if (size <= 0) {
    fclose(fp);
    std::cerr << "Input file is empty: " << file_path << std::endl;
    return nullptr;
  }
  void* buf = malloc(static_cast<size_t>(size));
  if (buf == nullptr) {
    fclose(fp);
    std::cerr << "Failed to allocate memory for input file: " << file_path
              << std::endl;
    return nullptr;
  }
  const size_t read_size = fread(buf, 1, static_cast<size_t>(size), fp);
  fclose(fp);
  if (read_size != static_cast<size_t>(size)) {
    free(buf);
    std::cerr << "Failed to read input file: " << file_path << std::endl;
    return nullptr;
  }
  *file_size = static_cast<size_t>(size);
  return buf;
}

static std::string ModelBaseName(const std::string& model_path) {
  const size_t slash = model_path.find_last_of("/\\");
  std::string name =
      slash == std::string::npos ? model_path : model_path.substr(slash + 1);
  const size_t dot = name.find_last_of('.');
  if (dot != std::string::npos) {
    name.resize(dot);
  }
  return name;
}

int main(int argc, char* argv[]) {
  if (argc < 2 || argc > 3) {
    std::cerr << "Usage: " << (argc > 0 ? argv[0] : "aml_compiled_model_case")
              << " <model.tflite> [input_file]\n"
              << "  Omit input_file to fill every input with random bytes.\n"
              << "  Place libLiteRt.so, libLiteRtCompilerPlugin_Aml.so and\n"
              << "  libLiteRtDispatch_Aml.so in the current directory.\n";
    return 1;
  }

  const std::string model_path = argv[1];
  const char* input_file = argc == 3 ? argv[2] : nullptr;
  const std::string model_name = ModelBaseName(model_path);

  std::vector<litert::EnvironmentOptions::Option> env_options;
  env_options.emplace_back(
      litert::EnvironmentOptions::Tag::kDispatchLibraryDir, ".");
  env_options.emplace_back(
      litert::EnvironmentOptions::Tag::kCompilerPluginLibraryDir, ".");
  env_options.emplace_back(litert::EnvironmentOptions::Tag::kCompilerCacheDir,
                           "./cache");
  litert::EnvironmentOptions environment_options(env_options);

  LITERT_ASSIGN_OR_ABORT(auto env,
                         litert::Environment::Create(environment_options));
  LITERT_ASSIGN_OR_ABORT(litert::Options options, litert::Options::Create());
  LITERT_RETURN_IF_ERROR(
      options.SetHardwareAccelerators(litert::HwAccelerators::kNpu));
  LITERT_ASSIGN_OR_ABORT(auto& aml_options,
                         options.GetOptions<litert::aml::AmlOptions>());
  LITERT_ABORT_IF_ERROR(aml_options.SetModelPath("/data/vendor/nn/"));
  LITERT_ABORT_IF_ERROR(aml_options.SetModelName(model_name));

  std::cout << "Loading model: " << model_path << std::endl;
  LITERT_ASSIGN_OR_ABORT(
      auto compiled_model,
      litert::CompiledModel::Create(env, model_path, options));

  auto input_buffers_result = compiled_model.CreateInputBuffers();
  if (!input_buffers_result) {
    std::cerr << "CreateInputBuffers failed: "
              << input_buffers_result.Error().Message() << std::endl;
    return 1;
  }
  auto output_buffers_result = compiled_model.CreateOutputBuffers();
  if (!output_buffers_result) {
    std::cerr << "CreateOutputBuffers failed: "
              << output_buffers_result.Error().Message() << std::endl;
    return 1;
  }
  auto input_buffers = std::move(*input_buffers_result);
  auto output_buffers = std::move(*output_buffers_result);

  size_t file_size = 0;
  void* file_data = nullptr;
  if (input_file != nullptr) {
    file_data = ReadFile(input_file, &file_size);
    if (file_data == nullptr) {
      return 1;
    }
  } else {
    std::cout << "No input file; filling inputs with random bytes" << std::endl;
  }

  std::mt19937 rng(std::random_device{}());
  std::uniform_int_distribution<int> byte_dist(0, 255);
  for (size_t input_index = 0; input_index < input_buffers.size();
       ++input_index) {
    auto& buf = input_buffers[input_index];
    LITERT_ASSIGN_OR_ABORT(const size_t size, buf.Size());
    std::vector<uint8_t> input_data(size);
    if (file_data != nullptr) {
      if (file_size < size) {
        std::cerr << "Input file size (" << file_size
                  << ") is smaller than input[" << input_index << "] size ("
                  << size << ")" << std::endl;
        free(file_data);
        return 1;
      }
      memcpy(input_data.data(), file_data, size);
    } else {
      for (size_t i = 0; i < size; ++i) {
        input_data[i] = static_cast<uint8_t>(byte_dist(rng));
      }
    }
    LITERT_RETURN_IF_ERROR(buf.Write(
        std::span<const uint8_t>(input_data.data(), input_data.size())));
  }
  free(file_data);

  std::cout << "Running inference" << std::endl;
  LITERT_RETURN_IF_ERROR(compiled_model.Run(input_buffers, output_buffers));

  for (size_t i = 0; i < output_buffers.size(); ++i) {
    LITERT_ASSIGN_OR_ABORT(const size_t size, output_buffers[i].Size());
    std::cout << "output[" << i << "] size=" << size << std::endl;
  }
  std::cout << "inference done" << std::endl;
  return 0;
}
