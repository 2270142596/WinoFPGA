/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow/lite/micro/micro_profiler.h"
#include <stdio.h>
#include <cinttypes>
#include <cstdint>
#include <cstring>

#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_time.h"

namespace tflite {

uint32_t MicroProfiler::BeginEvent(const char* tag) {
  if (num_events_ == kMaxEvents) {
    num_events_ = 0;
  }

  tags_[num_events_] = tag;
  start_ticks_[num_events_] = GetCurrentTimeTicks();
  end_ticks_[num_events_] = start_ticks_[num_events_] - 1;
  return num_events_++;
}

void MicroProfiler::EndEvent(uint32_t event_handle) {
  TFLITE_DCHECK(event_handle < kMaxEvents);
  end_ticks_[event_handle] = GetCurrentTimeTicks();
}

uint32_t MicroProfiler::GetTotalTicks() const {
  int32_t ticks = 0;
  for (int i = 0; i < num_events_; ++i) {
    ticks += end_ticks_[i] - start_ticks_[i];
  }
  return ticks;
}

void MicroProfiler::Log() const {
#if !defined(TF_LITE_STRIP_ERROR_STRINGS)
  for (int i = 0; i < num_events_; ++i) {
    uint32_t ticks = end_ticks_[i] - start_ticks_[i];
    MicroPrintf("%s took %u ticks (%d ms).", tags_[i], ticks,
                TicksToMs(ticks));
  }
#endif
}

void MicroProfiler::LogCsv() const {
  int count[7]={0};
#if !defined(TF_LITE_STRIP_ERROR_STRINGS)
  MicroPrintf("\"Event\",\"Tag\",\"Ticks\"");
  for (int i = 0; i < num_events_; ++i) {
    uint32_t ticks = end_ticks_[i] - start_ticks_[i];
    MicroPrintf("%d,%s,%u", i, tags_[i], ticks);
    if(0 == strcmp("CONV_2D",tags_[i])){
      count[0]+=ticks;
    }
    if(0 == strcmp("DEPTHWISE_CONV_2D",tags_[i])){
      count[1]+=ticks;
    }
        if(0 == strcmp("RELU",tags_[i])){
      count[2]+=ticks;
    }
        if(0 == strcmp("ADD",tags_[i])){
      count[3]+=ticks;
    }
            if(0 == strcmp("MEAN",tags_[i])){
      count[4]+=ticks;
    }
            if(0 == strcmp("RESHAPE",tags_[i])){
      count[5]+=ticks;
    }
    count[6]+=ticks;
  }
  printf("CONV_2D:%d\n",count[0]);
  printf("DEPTHWISE_CONV_2D:%d\n",count[1]);
  printf("RELU:%d\n",count[2]);
  printf("ADD:%d\n",count[3]);
  printf("MEAN:%d\n",count[4]);
  printf("RESHAPE:%d\n",count[5]);
  printf("NEEDES_ALL:%d\n",count[0]+count[1]+count[2]+count[3]+count[4]+count[5]);
  printf("ALL:%d\n",count[6]);

#endif
}

void MicroProfiler::LogTicksPerTagCsv() {
#if !defined(TF_LITE_STRIP_ERROR_STRINGS)
  MicroPrintf(
      "\"Unique Tag\",\"Total ticks across all events with that tag.\"");
  int total_ticks = 0;
  for (int i = 0; i < num_events_; ++i) {
    uint32_t ticks = end_ticks_[i] - start_ticks_[i];
    TFLITE_DCHECK(tags_[i] != nullptr);
    int position = FindExistingOrNextPosition(tags_[i]);
    TFLITE_DCHECK(position >= 0);
    total_ticks_per_tag[position].tag = tags_[i];
    total_ticks_per_tag[position].ticks =
        total_ticks_per_tag[position].ticks + ticks;
    total_ticks += ticks;
  }

  for (int i = 0; i < num_events_; ++i) {
    TicksPerTag each_tag_entry = total_ticks_per_tag[i];
    if (each_tag_entry.tag == nullptr) {
      break;
    }
    MicroPrintf("%s, %d", each_tag_entry.tag, each_tag_entry.ticks);
  }
  MicroPrintf("total number of ticks, %d", total_ticks);
#endif
}

// This method finds a particular array element in the total_ticks_per_tag array
// with the matching tag_name passed in the method. If it can find a
// matching array element that has the same tag_name, then it will return the
// position of the matching element. But if it unable to find a matching element
// with the given tag_name, it will return the next available empty position
// from the array.
int MicroProfiler::FindExistingOrNextPosition(const char* tag_name) {
  int pos = 0;
  for (; pos < num_events_; pos++) {
    TicksPerTag each_tag_entry = total_ticks_per_tag[pos];
    if (each_tag_entry.tag == nullptr ||
        strcmp(each_tag_entry.tag, tag_name) == 0) {
      return pos;
    }
  }
  return pos < num_events_ ? pos : -1;
}
}  // namespace tflite
