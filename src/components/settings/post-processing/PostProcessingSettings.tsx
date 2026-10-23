import React, { useEffect, useState } from "react";
import { Trans, useTranslation } from "react-i18next";
import { Plus, RefreshCcw, Trash2 } from "lucide-react";
import {
  commands,
  type PostProcessApiFormat,
  type PostProcessMode,
  type ThinkingLevel,
} from "@/bindings";

import { Alert } from "../../ui/Alert";
import {
  Dropdown,
  SettingContainer,
  SettingsGroup,
  Textarea,
} from "@/components/ui";
import { Button } from "../../ui/Button";
import { ResetButton } from "../../ui/ResetButton";
import { Input } from "../../ui/Input";

import { ProviderSelect } from "../PostProcessingSettingsApi/ProviderSelect";
import { BaseUrlField } from "../PostProcessingSettingsApi/BaseUrlField";
import { ApiKeyField } from "../PostProcessingSettingsApi/ApiKeyField";
import { ModelSelect } from "../PostProcessingSettingsApi/ModelSelect";
import { usePostProcessProviderState } from "../PostProcessingSettingsApi/usePostProcessProviderState";
import { useSettings } from "../../../hooks/useSettings";
import { PostProcessingContextToggle } from "../PostProcessingContextToggle";
import { PostProcessingToggle } from "../PostProcessingToggle";
import { ShortcutInput } from "../ShortcutInput";

const PostProcessingSettingsTriggerComponent: React.FC = () => {
  const { t } = useTranslation();
  const { getSetting, updateSetting, isUpdating } = useSettings();

  const mode = getSetting("post_process_mode") ?? "always";
  const isModeUpdating = isUpdating("post_process_mode");

  const thinking = getSetting("post_process_thinking") ?? "none";
  const isThinkingUpdating = isUpdating("post_process_thinking");

  const thinkingOptions: {
    value: ThinkingLevel;
    label: string;
    description: string;
  }[] = [
    {
      value: "none",
      label: t("settings.postProcessing.thinking.options.none"),
      description: t("settings.postProcessing.thinking.descriptions.none"),
    },
    {
      value: "low",
      label: t("settings.postProcessing.thinking.options.low"),
      description: t("settings.postProcessing.thinking.descriptions.low"),
    },
    {
      value: "medium",
      label: t("settings.postProcessing.thinking.options.medium"),
      description: t("settings.postProcessing.thinking.descriptions.medium"),
    },
    {
      value: "high",
      label: t("settings.postProcessing.thinking.options.high"),
      description: t("settings.postProcessing.thinking.descriptions.high"),
    },
  ];

  const handleModeChange = (value: PostProcessMode) => {
    updateSetting("post_process_mode", value);
  };

  const handleThinkingChange = (value: ThinkingLevel) => {
    updateSetting("post_process_thinking", value);
  };

  return (
    <>
      <PostProcessingToggle descriptionMode="tooltip" grouped={true} />

      <SettingContainer
        title={t("settings.postProcessing.trigger.mode.title")}
        description={t("settings.postProcessing.trigger.mode.description")}
        descriptionMode="tooltip"
        layout="stacked"
        grouped={true}
      >
        <div className="flex flex-col gap-2">
          <label className="flex items-center gap-2 text-sm cursor-pointer">
            <input
              type="radio"
              name="post_process_mode"
              checked={mode === "always"}
              onChange={() => handleModeChange("always")}
              disabled={isModeUpdating}
            />
            {t("settings.postProcessing.trigger.mode.always")}
          </label>
          <label className="flex items-center gap-2 text-sm cursor-pointer">
            <input
              type="radio"
              name="post_process_mode"
              checked={mode === "separate"}
              onChange={() => handleModeChange("separate")}
              disabled={isModeUpdating}
            />
            {t("settings.postProcessing.trigger.mode.separate")}
          </label>
          {mode === "always" && (
            <p className="text-xs text-mid-gray/70">
              {t("settings.postProcessing.trigger.alwaysNote")}
            </p>
          )}
        </div>
      </SettingContainer>

      {mode === "separate" && (
        <ShortcutInput
          shortcutId="transcribe_with_post_process"
          descriptionMode="tooltip"
          grouped={true}
        />
      )}

      <SettingContainer
        title={t("settings.postProcessing.thinking.title")}
        description={t("settings.postProcessing.thinking.description")}
        descriptionMode="tooltip"
        layout="stacked"
        grouped={true}
      >
        <div className="flex flex-col gap-2">
          <div
            role="radiogroup"
            aria-label={t("settings.postProcessing.thinking.title")}
            className="grid grid-cols-2 gap-2 sm:grid-cols-4"
          >
            {thinkingOptions.map((option) => (
              <button
                key={option.value}
                type="button"
                role="radio"
                aria-checked={thinking === option.value}
                disabled={isThinkingUpdating}
                onClick={() => handleThinkingChange(option.value)}
                className={`flex flex-col items-start gap-0.5 rounded-md border px-3 py-2 text-start transition-colors duration-150 cursor-pointer disabled:cursor-not-allowed disabled:opacity-50 ${
                  thinking === option.value
                    ? "border-logo-primary bg-logo-primary/20"
                    : "border-mid-gray/20 bg-mid-gray/10 hover:border-logo-primary hover:bg-logo-primary/10"
                }`}
              >
                <span className="text-sm font-semibold">{option.label}</span>
                <span className="text-xs text-mid-gray/70">
                  {option.description}
                </span>
              </button>
            ))}
          </div>
          <p className="text-xs text-mid-gray/70">
            {t("settings.postProcessing.thinking.deepseekNote")}
          </p>
        </div>
      </SettingContainer>
    </>
  );
};

const PostProcessingSettingsTrigger = React.memo(
  PostProcessingSettingsTriggerComponent,
);
PostProcessingSettingsTrigger.displayName = "PostProcessingSettingsTrigger";

const PostProcessingSettingsApiComponent: React.FC = () => {
  const { t } = useTranslation();
  const state = usePostProcessProviderState();
  const [isDeleting, setIsDeleting] = useState(false);

  const handleDeleteProvider = async () => {
    if (isDeleting) return;
    setIsDeleting(true);
    try {
      await state.handleDeleteProvider(state.selectedProviderId);
    } finally {
      setIsDeleting(false);
    }
  };

  return (
    <>
      <SettingContainer
        title={t("settings.postProcessing.api.provider.title")}
        description={t("settings.postProcessing.api.provider.description")}
        descriptionMode="tooltip"
        layout="horizontal"
        grouped={true}
      >
        <div className="flex flex-col gap-1">
          <div className="flex items-center gap-2">
            <ProviderSelect
              options={state.providerOptions}
              value={state.selectedProviderId}
              onChange={state.handleProviderSelect}
            />
            <Button
              onClick={state.handleAddProvider}
              variant="primary"
              size="md"
              className="shrink-0"
            >
              <span className="flex items-center gap-1.5">
                <Plus className="h-4 w-4" />
                {t("settings.postProcessing.api.provider.add")}
              </span>
            </Button>
            {state.isDeletableProvider && (
              <Button
                onClick={() => void handleDeleteProvider()}
                variant="secondary"
                size="md"
                disabled={isDeleting}
                className="shrink-0"
              >
                <span className="flex items-center gap-1.5">
                  <Trash2 className="h-4 w-4" />
                  {t("settings.postProcessing.api.provider.delete")}
                </span>
              </Button>
            )}
          </div>
          {!state.isAppleProvider && !state.isSelectedProviderPersisted && (
            <p className="text-xs text-mid-gray/70">
              {t("settings.postProcessing.api.provider.unsaved")}
            </p>
          )}
        </div>
      </SettingContainer>

      {state.isAppleProvider ? (
        state.appleIntelligenceUnavailable ? (
          <Alert variant="error" contained>
            {t("settings.postProcessing.api.appleIntelligence.unavailable")}
          </Alert>
        ) : null
      ) : (
        <>
          {state.isUserCustomProvider && (
            <SettingContainer
              title={t("settings.postProcessing.api.apiFormat.title")}
              description={t(
                "settings.postProcessing.api.apiFormat.description",
              )}
              descriptionMode="tooltip"
              layout="horizontal"
              grouped={true}
            >
              <Dropdown
                options={[
                  {
                    value: "openai",
                    label: t(
                      "settings.postProcessing.api.apiFormat.options.openai",
                    ),
                  },
                  {
                    value: "anthropic",
                    label: t(
                      "settings.postProcessing.api.apiFormat.options.anthropic",
                    ),
                  },
                ]}
                selectedValue={state.apiFormat}
                onSelect={(value) =>
                  state.handleApiFormatChange(value as PostProcessApiFormat)
                }
                disabled={state.isApiFormatUpdating}
              />
            </SettingContainer>
          )}

          <SettingContainer
            title={t("settings.postProcessing.api.baseUrl.title")}
            description={t("settings.postProcessing.api.baseUrl.description")}
            descriptionMode="tooltip"
            layout="horizontal"
            grouped={true}
          >
            <div className="flex items-center gap-2">
              <BaseUrlField
                value={state.baseUrl}
                onBlur={state.handleBaseUrlChange}
                placeholder={
                  state.apiFormat === "anthropic"
                    ? t(
                        "settings.postProcessing.api.baseUrl.placeholderAnthropic",
                      )
                    : t("settings.postProcessing.api.baseUrl.placeholder")
                }
                disabled={state.isBaseUrlUpdating || !state.allowBaseUrlEdit}
                className="min-w-[380px]"
              />
            </div>
          </SettingContainer>

          <SettingContainer
            title={t("settings.postProcessing.api.apiKey.title")}
            description={t("settings.postProcessing.api.apiKey.description")}
            descriptionMode="tooltip"
            layout="horizontal"
            grouped={true}
          >
            <div className="flex items-center gap-2">
              <ApiKeyField
                value={state.apiKey}
                onBlur={state.handleApiKeyChange}
                placeholder={t(
                  "settings.postProcessing.api.apiKey.placeholder",
                )}
                disabled={state.isApiKeyUpdating}
                className="min-w-[320px]"
              />
            </div>
          </SettingContainer>
        </>
      )}

      {!state.isAppleProvider && (
        <SettingContainer
          title={t("settings.postProcessing.api.model.title")}
          description={
            state.isUserCustomProvider
              ? t("settings.postProcessing.api.model.descriptionCustom")
              : t("settings.postProcessing.api.model.descriptionDefault")
          }
          descriptionMode="tooltip"
          layout="stacked"
          grouped={true}
        >
          <div className="flex items-center gap-2">
            <ModelSelect
              value={state.model}
              options={state.modelOptions}
              disabled={state.isModelUpdating}
              isLoading={state.isFetchingModels}
              placeholder={
                state.modelOptions.length > 0
                  ? t(
                      "settings.postProcessing.api.model.placeholderWithOptions",
                    )
                  : t("settings.postProcessing.api.model.placeholderNoOptions")
              }
              onSelect={state.handleModelSelect}
              onCreate={state.handleModelCreate}
              onBlur={() => {}}
              className="flex-1 min-w-[380px]"
            />
            <ResetButton
              onClick={state.handleRefreshModels}
              disabled={state.isFetchingModels}
              ariaLabel={t("settings.postProcessing.api.model.refreshModels")}
              className="flex h-10 w-10 items-center justify-center"
            >
              <RefreshCcw
                className={`h-4 w-4 ${state.isFetchingModels ? "animate-spin" : ""}`}
              />
            </ResetButton>
          </div>
        </SettingContainer>
      )}
    </>
  );
};

const PostProcessingSettingsPromptsComponent: React.FC = () => {
  const { t } = useTranslation();
  const { getSetting, updateSetting, isUpdating, refreshSettings } =
    useSettings();
  const [isCreating, setIsCreating] = useState(false);
  const [draftName, setDraftName] = useState("");
  const [draftText, setDraftText] = useState("");

  const prompts = getSetting("post_process_prompts") || [];
  const selectedPromptId = getSetting("post_process_selected_prompt_id") || "";
  const selectedPrompt =
    prompts.find((prompt) => prompt.id === selectedPromptId) || null;

  useEffect(() => {
    if (isCreating) return;

    if (selectedPrompt) {
      setDraftName(selectedPrompt.name);
      setDraftText(selectedPrompt.prompt);
    } else {
      setDraftName("");
      setDraftText("");
    }
  }, [
    isCreating,
    selectedPromptId,
    selectedPrompt?.name,
    selectedPrompt?.prompt,
  ]);

  const handlePromptSelect = (promptId: string | null) => {
    if (!promptId) return;
    updateSetting("post_process_selected_prompt_id", promptId);
    setIsCreating(false);
  };

  const handleCreatePrompt = async () => {
    if (!draftName.trim() || !draftText.trim()) return;

    try {
      const result = await commands.addPostProcessPrompt(
        draftName.trim(),
        draftText.trim(),
      );
      if (result.status === "ok") {
        await refreshSettings();
        updateSetting("post_process_selected_prompt_id", result.data.id);
        setIsCreating(false);
      }
    } catch (error) {
      console.error("Failed to create prompt:", error);
    }
  };

  const handleUpdatePrompt = async () => {
    if (!selectedPromptId || !draftName.trim() || !draftText.trim()) return;

    try {
      await commands.updatePostProcessPrompt(
        selectedPromptId,
        draftName.trim(),
        draftText.trim(),
      );
      await refreshSettings();
    } catch (error) {
      console.error("Failed to update prompt:", error);
    }
  };

  const handleDeletePrompt = async (promptId: string) => {
    if (!promptId) return;

    try {
      await commands.deletePostProcessPrompt(promptId);
      await refreshSettings();
      setIsCreating(false);
    } catch (error) {
      console.error("Failed to delete prompt:", error);
    }
  };

  const handleCancelCreate = () => {
    setIsCreating(false);
    if (selectedPrompt) {
      setDraftName(selectedPrompt.name);
      setDraftText(selectedPrompt.prompt);
    } else {
      setDraftName("");
      setDraftText("");
    }
  };

  const handleStartCreate = () => {
    setIsCreating(true);
    setDraftName("");
    setDraftText("");
  };

  const hasPrompts = prompts.length > 0;
  const isDirty =
    !!selectedPrompt &&
    (draftName.trim() !== selectedPrompt.name ||
      draftText.trim() !== selectedPrompt.prompt.trim());

  return (
    <SettingContainer
      title={t("settings.postProcessing.prompts.selectedPrompt.title")}
      description={t(
        "settings.postProcessing.prompts.selectedPrompt.description",
      )}
      descriptionMode="tooltip"
      layout="stacked"
      grouped={true}
    >
      <div className="space-y-3">
        <div className="flex gap-2 min-w-0">
          <Dropdown
            selectedValue={selectedPromptId || null}
            options={prompts.map((p) => ({
              value: p.id,
              label: p.name,
            }))}
            onSelect={(value) => handlePromptSelect(value)}
            placeholder={
              prompts.length === 0
                ? t("settings.postProcessing.prompts.noPrompts")
                : t("settings.postProcessing.prompts.selectPrompt")
            }
            disabled={
              isUpdating("post_process_selected_prompt_id") || isCreating
            }
            className="flex-1 min-w-0"
          />
          <Button
            onClick={handleStartCreate}
            variant="primary"
            size="md"
            disabled={isCreating}
            className="shrink-0"
          >
            {t("settings.postProcessing.prompts.createNew")}
          </Button>
        </div>

        {!isCreating && hasPrompts && selectedPrompt && (
          <div className="space-y-3">
            <div className="space-y-2 flex flex-col">
              <label className="text-sm font-semibold">
                {t("settings.postProcessing.prompts.promptLabel")}
              </label>
              <Input
                type="text"
                value={draftName}
                onChange={(e) => setDraftName(e.target.value)}
                placeholder={t(
                  "settings.postProcessing.prompts.promptLabelPlaceholder",
                )}
                variant="compact"
              />
            </div>

            <div className="space-y-2 flex flex-col">
              <label className="text-sm font-semibold">
                {t("settings.postProcessing.prompts.promptInstructions")}
              </label>
              <Textarea
                value={draftText}
                onChange={(e) => setDraftText(e.target.value)}
                placeholder={t(
                  "settings.postProcessing.prompts.promptInstructionsPlaceholder",
                )}
              />
              <p className="text-xs text-mid-gray/70">
                <Trans
                  i18nKey="settings.postProcessing.prompts.promptTip"
                  components={{ code: <code /> }}
                />
              </p>
            </div>

            <div className="flex gap-2 pt-2">
              <Button
                onClick={handleUpdatePrompt}
                variant="primary"
                size="md"
                disabled={!draftName.trim() || !draftText.trim() || !isDirty}
              >
                {t("settings.postProcessing.prompts.updatePrompt")}
              </Button>
              <Button
                onClick={() => handleDeletePrompt(selectedPromptId)}
                variant="secondary"
                size="md"
                disabled={!selectedPromptId || prompts.length <= 1}
              >
                {t("settings.postProcessing.prompts.deletePrompt")}
              </Button>
            </div>
          </div>
        )}

        {!isCreating && !selectedPrompt && (
          <div className="p-3 bg-mid-gray/5 rounded-md border border-mid-gray/20">
            <p className="text-sm text-mid-gray">
              {hasPrompts
                ? t("settings.postProcessing.prompts.selectToEdit")
                : t("settings.postProcessing.prompts.createFirst")}
            </p>
          </div>
        )}

        {isCreating && (
          <div className="space-y-3">
            <div className="space-y-2 block flex flex-col">
              <label className="text-sm font-semibold text-text">
                {t("settings.postProcessing.prompts.promptLabel")}
              </label>
              <Input
                type="text"
                value={draftName}
                onChange={(e) => setDraftName(e.target.value)}
                placeholder={t(
                  "settings.postProcessing.prompts.promptLabelPlaceholder",
                )}
                variant="compact"
              />
            </div>

            <div className="space-y-2 flex flex-col">
              <label className="text-sm font-semibold">
                {t("settings.postProcessing.prompts.promptInstructions")}
              </label>
              <Textarea
                value={draftText}
                onChange={(e) => setDraftText(e.target.value)}
                placeholder={t(
                  "settings.postProcessing.prompts.promptInstructionsPlaceholder",
                )}
              />
              <p className="text-xs text-mid-gray/70">
                <Trans
                  i18nKey="settings.postProcessing.prompts.promptTip"
                  components={{ code: <code /> }}
                />
              </p>
            </div>

            <div className="flex gap-2 pt-2">
              <Button
                onClick={handleCreatePrompt}
                variant="primary"
                size="md"
                disabled={!draftName.trim() || !draftText.trim()}
              >
                {t("settings.postProcessing.prompts.createPrompt")}
              </Button>
              <Button
                onClick={handleCancelCreate}
                variant="secondary"
                size="md"
              >
                {t("settings.postProcessing.prompts.cancel")}
              </Button>
            </div>
          </div>
        )}
      </div>
    </SettingContainer>
  );
};

export const PostProcessingSettingsApi = React.memo(
  PostProcessingSettingsApiComponent,
);
PostProcessingSettingsApi.displayName = "PostProcessingSettingsApi";

export const PostProcessingSettingsPrompts = React.memo(
  PostProcessingSettingsPromptsComponent,
);
PostProcessingSettingsPrompts.displayName = "PostProcessingSettingsPrompts";

export const PostProcessingSettings: React.FC = () => {
  const { t } = useTranslation();

  return (
    <div className="max-w-3xl w-full mx-auto space-y-6">
      <SettingsGroup title={t("settings.postProcessing.trigger.title")}>
        <PostProcessingSettingsTrigger />
      </SettingsGroup>

      <SettingsGroup title={t("settings.postProcessing.api.title")}>
        <PostProcessingSettingsApi />
      </SettingsGroup>

      <SettingsGroup title={t("settings.postProcessing.context.title")}>
        <PostProcessingContextToggle />
      </SettingsGroup>

      <SettingsGroup title={t("settings.postProcessing.prompts.title")}>
        <PostProcessingSettingsPrompts />
      </SettingsGroup>
    </div>
  );
};
