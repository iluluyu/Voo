import { useCallback, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";
import { useSettings } from "../../../hooks/useSettings";
import {
  commands,
  type PostProcessApiFormat,
  type PostProcessProvider,
} from "@/bindings";
import type { ModelOption } from "./types";
import type { DropdownOption } from "../../ui/Dropdown";

type PostProcessProviderState = {
  providerOptions: DropdownOption[];
  selectedProviderId: string;
  selectedProvider: PostProcessProvider | undefined;
  isUserCustomProvider: boolean;
  isAppleProvider: boolean;
  appleIntelligenceUnavailable: boolean;
  allowBaseUrlEdit: boolean;
  isDeletableProvider: boolean;
  isSelectedProviderPersisted: boolean;
  baseUrl: string;
  handleBaseUrlChange: (value: string) => void;
  isBaseUrlUpdating: boolean;
  apiFormat: PostProcessApiFormat;
  handleApiFormatChange: (value: PostProcessApiFormat) => void;
  isApiFormatUpdating: boolean;
  apiKey: string;
  handleApiKeyChange: (value: string) => void;
  isApiKeyUpdating: boolean;
  model: string;
  handleModelChange: (value: string) => void;
  modelOptions: ModelOption[];
  isModelUpdating: boolean;
  isFetchingModels: boolean;
  handleProviderSelect: (providerId: string) => void;
  handleAddProvider: () => string | null;
  handleDeleteProvider: (providerId: string) => Promise<void>;
  handleModelSelect: (value: string) => void;
  handleModelCreate: (value: string) => void;
  handleRefreshModels: () => void;
};

const APPLE_PROVIDER_ID = "apple_intelligence";
const LEGACY_CUSTOM_PROVIDER_ID = "custom";

// Mirrors MAX_CUSTOM_POST_PROCESS_PROVIDERS in src-tauri/src/settings.rs. The
// backend constant guards add_custom_post_process_provider, which is not
// exposed as a command yet, so this frontend guard is the active one.
const MAX_CUSTOM_PROVIDERS = 20;

/**
 * Dropdown order for built-in providers: DeepSeek first (out-of-the-box
 * default), 智谱 Coding Plan second, then the remaining built-ins in
 * default_post_process_providers() order with the legacy `custom` slot
 * closing the block. User-created `custom-N` entries render after all
 * built-ins, in creation order.
 */
const BUILTIN_PROVIDER_ORDER = [
  "deepseek",
  "zhipu-coding",
  "openai",
  "zai",
  "openrouter",
  "anthropic",
  "groq",
  "cerebras",
  APPLE_PROVIDER_ID,
  "bedrock_mantle",
  LEGACY_CUSTOM_PROVIDER_ID,
] as const;

const BUILTIN_PROVIDER_RANK = new Map<string, number>(
  BUILTIN_PROVIDER_ORDER.map((id, index) => [id, index]),
);

/** `custom-N` (N ≥ 1) provider number, or null for everything else. Mirrors
 * is_numbered_custom_provider_id in settings.rs — the legacy `custom` slot is
 * intentionally not numbered, so it can never be deleted. */
function numberedCustomProviderNumber(providerId: string): number | null {
  const match = /^custom-(\d+)$/.exec(providerId);
  if (!match) return null;
  const number = Number.parseInt(match[1], 10);
  return Number.isInteger(number) && number > 0 ? number : null;
}

/** True for every user-editable provider: the legacy `custom` slot plus any
 * user-created `custom-N` entry. Mirrors is_user_custom_provider_id. */
function isUserCustomProviderId(providerId: string): boolean {
  return (
    providerId === LEGACY_CUSTOM_PROVIDER_ID ||
    numberedCustomProviderNumber(providerId) !== null
  );
}

/** True only for user-created `custom-N` entries (deletable). Built-ins,
 * including the legacy `custom` slot, are never deletable. */
function isDeletableProviderId(providerId: string): boolean {
  return numberedCustomProviderNumber(providerId) !== null;
}

type SessionFieldValues = {
  baseUrl?: string;
  apiFormat?: PostProcessApiFormat;
  apiKey?: string;
  model?: string;
};

type ProviderSessionOverlay = {
  added: PostProcessProvider[];
  removedIds: Set<string>;
  fields: Map<string, SessionFieldValues>;
  selectedId: string | null;
};

/**
 * Session-scoped overlay for provider management. The backend data model
 * (settings.rs) supports custom-1…N plus add/remove helpers, but no Tauri
 * commands expose them yet, so the frontend maintains the array here: entries
 * added this session, persisted entries deleted this session (masked, since
 * nothing can remove them from the store), field values the existing
 * per-provider commands cannot persist, and the in-session selection.
 * Everything the existing commands CAN persist is still written through:
 * api key / model / selection for any backend-known id, and base URL / api
 * format for the legacy `custom` slot.
 */
const sessionOverlay: ProviderSessionOverlay = {
  added: [],
  removedIds: new Set<string>(),
  fields: new Map<string, SessionFieldValues>(),
  selectedId: null,
};

const snapshotOverlay = (): ProviderSessionOverlay => ({
  added: [...sessionOverlay.added],
  removedIds: new Set(sessionOverlay.removedIds),
  fields: new Map(sessionOverlay.fields),
  selectedId: sessionOverlay.selectedId,
});

export const usePostProcessProviderState = (): PostProcessProviderState => {
  const { t } = useTranslation();
  const {
    settings,
    isUpdating,
    setPostProcessProvider,
    updatePostProcessBaseUrl,
    updatePostProcessApiFormat,
    updatePostProcessApiKey,
    updatePostProcessModel,
    fetchPostProcessModels,
    postProcessModelOptions,
  } = useSettings();

  // Mirror of the module-level overlay; mutations go through mutateOverlay so
  // every hook instance (and remounts of the settings pane) stay in sync.
  const [overlay, setOverlay] = useState<ProviderSessionOverlay>(
    snapshotOverlay,
  );

  const mutateOverlay = useCallback(
    (mutate: (draft: ProviderSessionOverlay) => void) => {
      mutate(sessionOverlay);
      setOverlay(snapshotOverlay());
    },
    [],
  );

  const setSessionField = useCallback(
    (providerId: string, values: SessionFieldValues) => {
      mutateOverlay((draft) => {
        draft.fields.set(providerId, {
          ...draft.fields.get(providerId),
          ...values,
        });
      });
    },
    [mutateOverlay],
  );

  // Settings are guaranteed to have providers after migration
  const backendProviders = useMemo(() => {
    return settings?.post_process_providers ?? [];
  }, [settings?.post_process_providers]);

  const backendKnows = useCallback(
    (providerId: string) => {
      return backendProviders.some((provider) => provider.id === providerId);
    },
    [backendProviders],
  );

  // Merged view the dropdown renders: backend providers minus session-deleted
  // ones, plus session-added entries, ordered deepseek → zhipu-coding → other
  // built-ins → custom-N entries in creation order.
  const providers = useMemo<PostProcessProvider[]>(() => {
    const merged = backendProviders.filter(
      (provider) => !overlay.removedIds.has(provider.id),
    );
    for (const added of overlay.added) {
      if (!merged.some((provider) => provider.id === added.id)) {
        merged.push(added);
      }
    }
    return merged
      .map((provider, index) => ({
        provider,
        index,
        rank:
          BUILTIN_PROVIDER_RANK.get(provider.id) ?? BUILTIN_PROVIDER_ORDER.length,
      }))
      .sort((a, b) => a.rank - b.rank || a.index - b.index)
      .map((entry) => entry.provider);
  }, [backendProviders, overlay]);

  const selectedProviderId = useMemo(() => {
    if (
      overlay.selectedId &&
      providers.some((provider) => provider.id === overlay.selectedId)
    ) {
      return overlay.selectedId;
    }
    const storedId = settings?.post_process_provider_id;
    if (storedId && providers.some((provider) => provider.id === storedId)) {
      return storedId;
    }
    // DeepSeek is the first entry after ordering and the out-of-the-box
    // default; fall back to it whenever nothing valid is stored/selected.
    return providers[0]?.id ?? "deepseek";
  }, [overlay.selectedId, providers, settings?.post_process_provider_id]);

  const selectedProvider = useMemo(() => {
    return (
      providers.find((provider) => provider.id === selectedProviderId) ||
      providers[0]
    );
  }, [providers, selectedProviderId]);

  const isAppleProvider = selectedProvider?.id === APPLE_PROVIDER_ID;
  const [appleIntelligenceUnavailable, setAppleIntelligenceUnavailable] =
    useState(false);

  // Session overlay first (values the backend commands cannot persist), then
  // the persisted settings as the single source of truth.
  const sessionFields = overlay.fields.get(selectedProviderId);
  const baseUrl = sessionFields?.baseUrl ?? selectedProvider?.base_url ?? "";
  const apiFormat =
    sessionFields?.apiFormat ?? selectedProvider?.api_format ?? "openai";
  const apiKey =
    sessionFields?.apiKey ??
    settings?.post_process_api_keys?.[selectedProviderId] ??
    "";
  const model =
    sessionFields?.model ??
    settings?.post_process_models?.[selectedProviderId] ??
    "";

  const providerOptions = useMemo<DropdownOption[]>(() => {
    return providers.map((provider) => ({
      value: provider.id,
      label: provider.label,
    }));
  }, [providers]);

  const isUserCustomProvider = isUserCustomProviderId(selectedProviderId);
  const allowBaseUrlEdit = selectedProvider
    ? selectedProvider.allow_base_url_edit !== false
    : false;
  const isDeletableProvider = isDeletableProviderId(selectedProviderId);
  const isSelectedProviderPersisted = backendKnows(selectedProviderId);

  const handleProviderSelect = useCallback(
    async (providerId: string) => {
      // Clear error state on any selection attempt (allows dismissing the error)
      setAppleIntelligenceUnavailable(false);

      if (providerId === selectedProviderId) return;

      // Check Apple Intelligence availability before selecting
      if (providerId === APPLE_PROVIDER_ID) {
        const available = await commands.checkAppleIntelligenceAvailable();
        if (!available) {
          setAppleIntelligenceUnavailable(true);
          // Don't return - still set the provider so dropdown shows the selection
          // The backend gracefully handles unavailable Apple Intelligence
        }
      }

      // Record the selection in the session overlay first so entries that
      // only exist client-side can be selected too; persist through the
      // existing command whenever the backend knows the id.
      mutateOverlay((draft) => {
        draft.selectedId = providerId;
      });
      if (backendKnows(providerId)) {
        await setPostProcessProvider(providerId);
      }

      // Auto-fetch available models for the new provider so the model dropdown
      // reflects what's actually valid. Without this, a stale model value from
      // a previous provider/base_url can persist and silently 404 at runtime.
      // Skip when the provider isn't configured yet (no API key / empty base URL)
      // or only exists in the session overlay (the backend can't resolve it).
      if (providerId !== APPLE_PROVIDER_ID && backendKnows(providerId)) {
        const provider = providers.find((p) => p.id === providerId);
        const apiKey = settings?.post_process_api_keys?.[providerId] ?? "";
        const hasBaseUrl = (provider?.base_url ?? "").trim() !== "";
        const hasApiKey = apiKey.trim() !== "";

        if (isUserCustomProviderId(providerId) ? hasBaseUrl : hasApiKey) {
          void fetchPostProcessModels(providerId);
        }
      }
    },
    [
      selectedProviderId,
      backendKnows,
      mutateOverlay,
      setPostProcessProvider,
      providers,
      settings,
      fetchPostProcessModels,
    ],
  );

  const handleAddProvider = useCallback((): string | null => {
    const customCount = providers.filter((provider) =>
      isDeletableProviderId(provider.id),
    ).length;
    if (customCount >= MAX_CUSTOM_PROVIDERS) {
      toast.error(
        t("settings.postProcessing.api.provider.limitReached", {
          count: MAX_CUSTOM_PROVIDERS,
        }),
      );
      return null;
    }

    // Smallest unused positive number keeps ids short (mirrors the backend
    // add helper) and lets a deleted id be reused within the session.
    let number = 1;
    while (providers.some((provider) => provider.id === `custom-${number}`)) {
      number += 1;
    }

    const provider: PostProcessProvider = {
      id: `custom-${number}`,
      label: t("settings.postProcessing.api.provider.defaultLabel", { number }),
      base_url: "",
      allow_base_url_edit: true,
      models_endpoint: "/models",
      supports_structured_output: false,
      api_format: "openai",
    };

    mutateOverlay((draft) => {
      draft.added.push(provider);
      draft.removedIds.delete(provider.id);
      draft.selectedId = provider.id;
    });
    return provider.id;
  }, [providers, mutateOverlay, t]);

  const handleDeleteProvider = useCallback(
    async (providerId: string) => {
      if (!isDeletableProviderId(providerId)) return;
      const wasPersisted = backendKnows(providerId);
      const storedId = settings?.post_process_provider_id ?? null;

      // Clear the stored key and model through the existing per-provider
      // commands so no credentials linger (only possible for backend-known
      // ids; overlay-only entries never reached the store).
      if (wasPersisted) {
        await updatePostProcessApiKey(providerId, "");
        await updatePostProcessModel(providerId, "");
      }

      const remaining = providers.filter(
        (provider) => provider.id !== providerId,
      );
      const fallbackId = remaining[0]?.id ?? "deepseek";
      const storedStillValid =
        storedId !== null &&
        storedId !== providerId &&
        remaining.some((provider) => provider.id === storedId);

      mutateOverlay((draft) => {
        draft.added = draft.added.filter((p) => p.id !== providerId);
        draft.fields.delete(providerId);
        if (wasPersisted) {
          // No backend remove command exists: mask the persisted entry for
          // the rest of the session instead of dropping it from the store.
          draft.removedIds.add(providerId);
        }
        draft.selectedId = storedStillValid ? null : fallbackId;
      });

      // If the deleted provider was the persisted selection, repoint the
      // stored selection so it doesn't dangle in the backend.
      if (wasPersisted && storedId === providerId) {
        await setPostProcessProvider(fallbackId);
      }
    },
    [
      backendKnows,
      providers,
      mutateOverlay,
      settings?.post_process_provider_id,
      setPostProcessProvider,
      updatePostProcessApiKey,
      updatePostProcessModel,
    ],
  );

  const handleBaseUrlChange = useCallback(
    (value: string) => {
      if (!selectedProvider || !isUserCustomProviderId(selectedProvider.id)) {
        return;
      }
      const trimmed = value.trim();
      if (!trimmed || trimmed !== baseUrl) {
        // The base-url command only accepts the legacy `custom` slot today;
        // custom-N base URLs live in the session overlay until backend
        // provider-management commands ship.
        setSessionField(selectedProvider.id, { baseUrl: trimmed });
        if (selectedProvider.id === LEGACY_CUSTOM_PROVIDER_ID) {
          void updatePostProcessBaseUrl(selectedProvider.id, trimmed);
        }
      }
    },
    [selectedProvider, baseUrl, setSessionField, updatePostProcessBaseUrl],
  );

  const handleApiFormatChange = useCallback(
    (value: PostProcessApiFormat) => {
      if (value === apiFormat) return;
      if (isUserCustomProviderId(selectedProviderId)) {
        setSessionField(selectedProviderId, { apiFormat: value });
      }
      if (selectedProviderId === LEGACY_CUSTOM_PROVIDER_ID) {
        void updatePostProcessApiFormat(selectedProviderId, value);
      }
    },
    [
      apiFormat,
      selectedProviderId,
      setSessionField,
      updatePostProcessApiFormat,
    ],
  );

  const handleApiKeyChange = useCallback(
    (value: string) => {
      const trimmed = value.trim();
      if (trimmed === apiKey) return;
      if (backendKnows(selectedProviderId)) {
        void updatePostProcessApiKey(selectedProviderId, trimmed);
      } else {
        setSessionField(selectedProviderId, { apiKey: trimmed });
      }
    },
    [
      apiKey,
      selectedProviderId,
      backendKnows,
      setSessionField,
      updatePostProcessApiKey,
    ],
  );

  const applyModelValue = useCallback(
    (value: string) => {
      const trimmed = value.trim();
      if (trimmed === model) return;
      if (backendKnows(selectedProviderId)) {
        void updatePostProcessModel(selectedProviderId, trimmed);
      } else {
        setSessionField(selectedProviderId, { model: trimmed });
      }
    },
    [
      model,
      selectedProviderId,
      backendKnows,
      setSessionField,
      updatePostProcessModel,
    ],
  );

  const handleModelChange = applyModelValue;
  const handleModelSelect = applyModelValue;
  const handleModelCreate = applyModelValue;

  const handleRefreshModels = useCallback(() => {
    if (isAppleProvider) return;
    if (!backendKnows(selectedProviderId)) return;
    void fetchPostProcessModels(selectedProviderId);
  }, [isAppleProvider, backendKnows, selectedProviderId, fetchPostProcessModels]);

  const availableModelsRaw = postProcessModelOptions[selectedProviderId] || [];

  const modelOptions = useMemo<ModelOption[]>(() => {
    const seen = new Set<string>();
    const options: ModelOption[] = [];

    const upsert = (value: string | null | undefined) => {
      const trimmed = value?.trim();
      if (!trimmed || seen.has(trimmed)) return;
      seen.add(trimmed);
      options.push({ value: trimmed, label: trimmed });
    };

    // Add available models from API
    for (const candidate of availableModelsRaw) {
      upsert(candidate);
    }

    // Ensure current model is in the list
    upsert(model);

    return options;
  }, [availableModelsRaw, model]);

  const isBaseUrlUpdating = isUpdating(
    `post_process_base_url:${selectedProviderId}`,
  );
  const isApiFormatUpdating = isUpdating(
    `post_process_api_format:${selectedProviderId}`,
  );
  const isApiKeyUpdating = isUpdating(
    `post_process_api_key:${selectedProviderId}`,
  );
  const isModelUpdating = isUpdating(
    `post_process_model:${selectedProviderId}`,
  );
  const isFetchingModels = isUpdating(
    `post_process_models_fetch:${selectedProviderId}`,
  );

  // No automatic fetching - user must click refresh button

  return {
    providerOptions,
    selectedProviderId,
    selectedProvider,
    isUserCustomProvider,
    isAppleProvider,
    appleIntelligenceUnavailable,
    allowBaseUrlEdit,
    isDeletableProvider,
    isSelectedProviderPersisted,
    baseUrl,
    handleBaseUrlChange,
    isBaseUrlUpdating,
    apiFormat,
    handleApiFormatChange,
    isApiFormatUpdating,
    apiKey,
    handleApiKeyChange,
    isApiKeyUpdating,
    model,
    handleModelChange,
    modelOptions,
    isModelUpdating,
    isFetchingModels,
    handleProviderSelect,
    handleAddProvider,
    handleDeleteProvider,
    handleModelSelect,
    handleModelCreate,
    handleRefreshModels,
  };
};
