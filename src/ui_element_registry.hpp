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

#pragma once

/*
 * Element registry - data-driven description of the glTF collections shown in the
 * Scene Browser's "Elements" tab.
 *
 * Each glTF collection (nodes, meshes, materials, cameras, lights, textures, images,
 * samplers, animations) is described once by an ElementTypeDesc. One generic renderer
 * (UiSceneBrowser::renderElementsTab) draws the icon tab bar, the columnar table, the
 * Add / Duplicate / Delete / Rename toolbar and the inline rename for EVERY category
 * from these descriptors. Adding a new collection is one descriptor, no new UI code.
 *
 * These are pure type declarations (no logic): the descriptors are built - with lambdas
 * closing over the live Scene / SceneSelection / UndoStack - in ui_scene_browser_elements.cpp.
 */

#include <functional>
#include <string>
#include <vector>

#include "scene_selection.hpp"

// One column of the element table. `draw` renders the cell for element `index`; an optional
// `sortKey` (numeric) enables click-to-sort on that column (lexical for the name column).
struct ElementColumn
{
  const char*                      header     = "";
  float                            width      = 0.0f;  // 0 => stretch, >0 => fixed pixels
  bool                             rightAlign = false;
  std::function<void(int index)>   draw;
  std::function<double(int index)> sortKey;  // optional; nullptr => not sortable
};

// One entry in a category's Add menu (or its single default, e.g. an empty node). `create`
// performs the undoable creation and selects the new element.
struct ElementAddVariant
{
  const char*           label = "";
  const char*           icon  = "";
  std::function<void()> create;
};

// Describes one glTF collection for the generic Elements list. A null CRUD handler simply hides
// the corresponding toolbar button for that category, so the toolbar stays uniform but honest.
struct ElementTypeDesc
{
  const char*                   icon     = "";  // tab + row-fallback icon (ICON_MS_*)
  const char*                   singular = "";  // "Node"  - tooltips, toolbar
  const char*                   plural   = "";  // "Nodes" - tab tooltip, footer counts
  SceneSelection::SelectionType selKind  = SceneSelection::SelectionType::eNone;

  std::function<int()>            count;   // number of elements
  std::function<std::string(int)> name;    // display-only label (may be derived: URI, "Sampler N", ...)
  std::function<void(int)>        select;  // set the shared selection to this element

  // Optional override: return the list index [0, count) that corresponds to the current selection,
  // or -1 if nothing in this category is selected. When null, the default selKind-based lookup
  // in selectedElementIndex() is used. Needed for categories that mix multiple selection types
  // (e.g. KHR lights + IES-only nodes in one list).
  std::function<int(const SceneSelection::SelectionContext&, const tinygltf::Model&)> selectedIndexFor;

  // Raw, editable glTF name used to seed the rename text field. Optional: null => `name` is already the
  // raw field and doubles as the seed. Set this when `name` is a derived display label (e.g. an image's
  // URI/placeholder fallback) so a no-op rename can't write that derived text back as the real name.
  std::function<std::string(int)> editableName;

  std::vector<ElementColumn> columns;  // browse info; columns[0] is typically Name

  // CRUD - any null handler hides that action for this category.
  std::vector<ElementAddVariant>               addVariants;           // empty => no Add
  std::function<void(int)>                     duplicate;             // null => no Duplicate
  std::function<bool(int)>                     canDelete;             // null => deletable when remove set
  std::function<std::string(int)>              deleteDisabledReason;  // optional tooltip when !canDelete
  std::function<void(int)>                     remove;                // null => no Delete
  std::function<void(int, const std::string&)> rename;                // null => no Rename
  std::function<void(int)>                     contextMenuExtra;      // optional per-type extra menu items

  // Optional one-line aggregate shown after the count in the footer (e.g. total triangles / texture MB).
  std::function<std::string()> footerAggregate;
};
