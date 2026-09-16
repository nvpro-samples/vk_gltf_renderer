/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/*
 * One declaration per user-settable value.
 *
 * A setting has to reach three consumers: the command line, benchmark parameter sequences, and the
 * ImGui.ini that remembers it between runs. The first two read nvutils::ParameterRegistry; the last
 * reads nvgui::SettingsHandler.
 *
 * Those used to be two hand-maintained lists, registered at two different points in the lifecycle
 * -- parameters in the constructor, settings in onAttach -- so no single place showed both and they
 * drifted badly (the tonemapper persisted its method but not its exposure; ptTechnique persisted
 * but ptUseSER did not). Declaring through this registry makes that impossible: one call feeds them
 * all, and Persist is a decision you have to state rather than forget.
 *
 * Because the settings handler only exists later, persisted declarations are recorded here and
 * replayed by applyPersistence() once it does.
 *
 * ImGui.ini restore writes directly into the storage pointer through nvgui::SettingsHandler and
 * does NOT fire the parameter's callbackSuccess -- unlike the command-line and benchmark paths,
 * which do. Most callbacks are CLI/UI-only "one-shot" side effects that should NOT re-run on
 * restore (e.g. --ptSamples disables adaptive sampling, but restoring ptSamples from a session
 * where the user had adaptive sampling on must not clobber the equally-restored ptAdaptiveSampling).
 * For the few cases where a persisted value drives derived state that must be recomputed after
 * restore (e.g. skySunAzimuth/Elevation drive skyParams.sunDirection), callers explicitly opt in
 * with addPostRestoreHook() and we replay those hooks in runPostRestoreHooks() once ImGui has
 * finished loading the ini (see GltfRenderer::onUIRender).
 */

#include <functional>
#include <string>
#include <utility>
#include <vector>

#include <nvgui/settings_handler.hpp>
#include <nvutils/parameter_registry.hpp>

// Whether a value is remembered in ImGui.ini between runs.
//
// eNo is for anything consumed once at start-up (window size, device index, headless frame count)
// and for actions -- remembering "load this file" or "take a screenshot" would replay it on the
// next launch.
enum class Persist
{
  eNo,
  eYes
};

class SettingsRegistry
{
public:
  explicit SettingsRegistry(nvutils::ParameterRegistry* registry)
      : m_registry(registry)
  {
  }

  // Scalars, paths, and enums-as-int. Trailing arguments (min/max) forward to the registry.
  template <class T, class... Rest>
  void add(const nvutils::ParameterBase::Info& info, T* storage, Persist persist, Rest&&... rest)
  {
    m_registry->add(info, storage, std::forward<Rest>(rest)...);
    if(persist == Persist::eYes)
      remember(info.name, storage);
  }

  // glm vectors (colors, sizes).
  template <class GLMVEC>
  void addVector(const nvutils::ParameterBase::Info& info, GLMVEC* storage, Persist persist)
  {
    m_registry->addVector(info, storage);
    if(persist == Persist::eYes)
      remember(info.name, storage);
  }

  // An action: no storage, just a callback. Runs from the command line and a benchmark sequence
  // alike. callbackSuccess fires on the application thread, so the callback may touch scene and
  // Vulkan state. Never persisted.
  void addAction(nvutils::ParameterBase::Info info, std::function<void()> action)
  {
    info.callbackSuccess = [action = std::move(action)](const nvutils::ParameterBase* const) { action(); };
    m_actionStorage.push_back(std::make_unique<bool>(false));
    m_registry->add(info, m_actionStorage.back().get(), true);
  }

  // Replay the persisted declarations. Called once the handler exists (onAttach); declarations
  // happen earlier, in the constructor.
  void applyPersistence(nvgui::SettingsHandler& handler) const
  {
    for(size_t i = 0; i < m_persisted.size(); i++)
      m_persisted[i](handler);
  }

  // Register a hook that should run once after ImGui has finished loading the ini. Use this for
  // derived state that a persisted value drives but that ini restore does not itself refresh
  // (e.g. recompute skyParams.sunDirection from the restored skySunAzimuth/Elevation). Prefer
  // this over reusing ParameterBase::callbackSuccess, whose semantics are CLI/UI edits, not
  // full-session restore.
  //
  // CLI precedence contract: hooks must be *derived* and *idempotent*. They read persisted
  // storage after nvgui::SettingsHandler's loadFilter has already skipped keys marked
  // ParameterParser::wasParsed() (see GltfRenderer::onAttach), so the values a hook sees are
  // already "CLI wins per key, ini fills the rest". Never write back to a persisted storage
  // pointer from a hook -- that would silently overwrite a CLI-set value.
  void addPostRestoreHook(std::function<void()> hook) { m_postRestoreHooks.push_back(std::move(hook)); }

  // Fire all registered post-restore hooks. Call once after ImGui has loaded the ini
  // (i.e. after Application::run()'s ini reload, e.g. on the first frame).
  void runPostRestoreHooks() const
  {
    for(size_t i = 0; i < m_postRestoreHooks.size(); i++)
      m_postRestoreHooks[i]();
  }

  [[nodiscard]] const std::vector<std::string>& persistedNames() const { return m_persistedNames; }

private:
  template <class T>
  void remember(const std::string& name, T* storage)
  {
    m_persistedNames.push_back(name);
    m_persisted.push_back([name, storage](nvgui::SettingsHandler& handler) { handler.setSetting(name, storage); });
  }

  nvutils::ParameterRegistry*                               m_registry{};
  std::vector<std::function<void(nvgui::SettingsHandler&)>> m_persisted;
  std::vector<std::string>                                  m_persistedNames;
  std::vector<std::function<void()>>                        m_postRestoreHooks;
  // Flag parameters need somewhere to write; an action only cares that the callback fired.
  std::vector<std::unique_ptr<bool>> m_actionStorage;
};
