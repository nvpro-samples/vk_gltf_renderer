/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gltf_interactivity_pointer.hpp"
#include "gltf_interactivity_instance.hpp"

namespace nvvkgltf {

std::vector<PointerTemplateSegment> parsePointerTemplate(const std::string& templateStr)
{
  std::vector<PointerTemplateSegment> segments;
  std::string                         literal;

  size_t i = 0;
  while(i < templateStr.size())
  {
    const char c = templateStr[i];
    if(c == '[' || c == '{')
    {
      const char   closing = (c == '[') ? ']' : '}';
      const size_t end     = templateStr.find(closing, i + 1);
      if(end == std::string::npos)
      {
        // Malformed template (unclosed parameter): treat the rest as literal text rather than
        // throwing away the segments already parsed. Graphs with malformed templates simply fail
        // to resolve at runtime (substitutePointerTemplate can't find a socket named "[foo"), not
        // a hard error here.
        literal += templateStr.substr(i);
        break;
      }
      if(!literal.empty())
      {
        segments.push_back({PointerTemplateSegment::Kind::eLiteral, literal});
        literal.clear();
      }
      const std::string paramName = templateStr.substr(i + 1, end - i - 1);
      segments.push_back({c == '[' ? PointerTemplateSegment::Kind::eIntParam : PointerTemplateSegment::Kind::eRefParam, paramName});
      i = end + 1;
    }
    else
    {
      literal += c;
      ++i;
    }
  }
  if(!literal.empty())
    segments.push_back({PointerTemplateSegment::Kind::eLiteral, literal});
  return segments;
}

std::optional<std::string> substitutePointerTemplate(const std::vector<PointerTemplateSegment>& segments,
                                                     InteractivityGraphInstance&                instance,
                                                     const InteractivityNode&                   node)
{
  std::string result;
  for(const PointerTemplateSegment& seg : segments)
  {
    switch(seg.kind)
    {
      case PointerTemplateSegment::Kind::eLiteral:
        result += seg.text;
        break;
      case PointerTemplateSegment::Kind::eIntParam: {
        const InteractivityValue v = instance.evaluateInput(node, seg.text);
        if(!std::holds_alternative<int32_t>(v))
          return std::nullopt;
        result += std::to_string(std::get<int32_t>(v));
        break;
      }
      case PointerTemplateSegment::Kind::eRefParam: {
        const InteractivityValue v = instance.evaluateInput(node, seg.text);
        if(!std::holds_alternative<InteractivityRef>(v))
          return std::nullopt;
        result += std::to_string(std::get<InteractivityRef>(v).handle);
        break;
      }
    }
  }
  return result;
}

}  // namespace nvvkgltf
