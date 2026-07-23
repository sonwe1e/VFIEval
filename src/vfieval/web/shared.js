(function (root) {
  "use strict";

  function responseEnvelope(payload) {
    return payload && payload.error && typeof payload.error === "object"
      ? payload.error
      : {};
  }

  function responseHeader(response, name) {
    return response && response.headers && typeof response.headers.get === "function"
      ? response.headers.get(name)
      : "";
  }

  async function readResponse(response) {
    const contentType = String(responseHeader(response, "content-type") || "");
    const text = await response.text();
    if (!text.trim()) {
      return { data: {}, kind: "empty", contentType, text: "" };
    }
    try {
      return { data: JSON.parse(text), kind: "json", contentType, text };
    } catch (_error) {
      return {
        data: { message: text.trim(), raw_text: text },
        kind: "text",
        contentType,
        text,
      };
    }
  }

  function errorMessage(payload, response) {
    const envelope = responseEnvelope(payload);
    if (envelope.message) return String(envelope.message);
    if (payload && typeof payload.error === "string" && payload.error.trim()) {
      return payload.error.trim();
    }
    if (payload && payload.message) return String(payload.message);
    return String((response && response.statusText) || "请求失败");
  }

  function diagnosticIds(payload, response) {
    const envelope = responseEnvelope(payload);
    return {
      request_id: String(
        envelope.request_id
        || (payload && payload.request_id)
        || responseHeader(response, "X-Request-ID")
        || "",
      ),
      support_id: String(
        envelope.support_id
        || (payload && payload.support_id)
        || responseHeader(response, "X-Support-ID")
        || "",
      ),
    };
  }

  function recoverySuggestion(errorLike) {
    const source = errorLike || {};
    const status = Number(source.status || 0);
    const code = String(source.code || "").toLowerCase();
    const name = String(source.name || "").toLowerCase();
    if (code.indexOf("timeout") >= 0 || name.indexOf("timeout") >= 0) {
      return "请检查服务地址与网络连接后重试；若持续超时，请把诊断编号交给维护人员。";
    }
    if (code === "network_error" || source.network) {
      return "请确认 VFIEval 服务仍在运行、当前地址可访问，然后重试本操作。";
    }
    if (status === 401 || status === 403) {
      return "当前会话没有执行此操作的权限，请刷新会话或联系组织者。";
    }
    if (status === 404) {
      return "目标可能已被删除或链接已失效，请返回列表刷新后重试。";
    }
    if (status === 409) {
      return "当前状态或提交内容已变化，请刷新最新状态后再试；不要连续重复提交。";
    }
    if (status === 507) {
      return "请先清理旧产物或释放存储空间，再重新提交。";
    }
    if (status >= 500) {
      return "服务暂时无法完成请求，请稍后重试；若持续失败，请复制诊断编号。";
    }
    return "请核对当前输入并重试；若问题持续，请复制诊断编号交给维护人员。";
  }

  function createError(options) {
    const settings = options || {};
    const payload = settings.payload && typeof settings.payload === "object"
      ? settings.payload
      : {};
    const response = settings.response || null;
    const envelope = responseEnvelope(payload);
    const cause = settings.cause || null;
    const statusValue = settings.status !== undefined
      ? settings.status
      : (response && response.status !== undefined
        ? response.status
        : (cause && cause.status !== undefined ? cause.status : 0));
    const status = Number(statusValue || 0);
    const ids = diagnosticIds(payload, response);
    const code = String(
      settings.code
      || envelope.code
      || envelope.type
      || payload.code
      || (cause && cause.code)
      || (settings.network ? "network_error" : "")
      || "request_failed",
    );
    const message = String(
      settings.message
      || errorMessage(payload, response)
      || (cause && cause.message)
      || "请求失败",
    );
    let details = null;
    if (Object.prototype.hasOwnProperty.call(settings, "details")) details = settings.details;
    else if (Object.prototype.hasOwnProperty.call(envelope, "details")) details = envelope.details;
    else if (Object.prototype.hasOwnProperty.call(payload, "details")) details = payload.details;
    const error = new Error(message);
    if (settings.name) error.name = String(settings.name);
    error.code = code;
    error.status = status;
    error.payload = payload;
    error.request_id = ids.request_id;
    error.support_id = ids.support_id;
    error.details = details;
    error.recovery_suggestion = String(
      settings.recovery_suggestion
      || recoverySuggestion({
        status,
        code,
        name: error.name,
        network: Boolean(settings.network),
      }),
    );
    if (cause) error.cause = cause;
    return error;
  }

  function networkError(cause, fallbackMessage) {
    if (cause && cause.name === "AbortError") return cause;
    const causeMessage = String((cause && cause.message) || "");
    const generic = !causeMessage || /failed to fetch|networkerror|load failed/i.test(causeMessage);
    return createError({
      cause,
      code: "network_error",
      message: generic ? (fallbackMessage || "无法连接 VFIEval 服务") : causeMessage,
      network: true,
    });
  }

  function timeoutError(message) {
    return createError({
      code: "request_timeout",
      name: "TimeoutError",
      message: message || "请求超时，请稍后重试。",
    });
  }

  function reportDiagnostic(settings, error) {
    if (settings.suppressDiagnostic || typeof settings.onDiagnostic !== "function") return;
    try {
      settings.onDiagnostic(error);
    } catch (_error) {
      // Diagnostic presentation must never replace the original request error.
    }
  }

  async function request(path, options) {
    const settings = options || {};
    if (typeof root.fetch !== "function") {
      const unsupported = createError({
        code: "fetch_unavailable",
        message: settings.unsupportedMessage || "当前浏览器不支持所需的网络功能。",
        recovery_suggestion: "请升级浏览器后重新打开页面。",
      });
      reportDiagnostic(settings, unsupported);
      throw unsupported;
    }
    const fetchOptions = Object.assign({}, settings.fetchOptions || {});
    if (settings.defaultJsonHeader !== false) {
      fetchOptions.headers = Object.assign(
        { "Content-Type": "application/json" },
        fetchOptions.headers || {},
      );
    }
    let timeoutId = null;
    let response;
    try {
      const responsePromise = root.fetch(path, fetchOptions);
      if (Number(settings.timeoutMs || 0) > 0) {
        const timeout = new Promise(function (_resolve, reject) {
          timeoutId = root.setTimeout(function () {
            reject(timeoutError(settings.timeoutMessage));
          }, Number(settings.timeoutMs));
        });
        response = await Promise.race([responsePromise, timeout]);
      } else {
        response = await responsePromise;
      }
    } catch (cause) {
      if (cause && cause.name === "AbortError") throw cause;
      const error = cause && cause.recovery_suggestion
        ? cause
        : networkError(cause, settings.networkMessage || "无法连接 VFIEval 服务");
      reportDiagnostic(settings, error);
      throw error;
    } finally {
      if (timeoutId !== null) root.clearTimeout(timeoutId);
    }

    let parsed;
    try {
      parsed = await readResponse(response);
    } catch (cause) {
      if (cause && cause.name === "AbortError") throw cause;
      const error = networkError(cause, settings.networkMessage || "无法读取 VFIEval 响应");
      reportDiagnostic(settings, error);
      throw error;
    }
    const invalidJson = parsed.kind !== "json"
      && parsed.contentType.indexOf("application/json") >= 0;
    const data = invalidJson
      ? { error: { message: settings.invalidJsonMessage || "服务器返回了无法解析的 JSON 响应。" } }
      : parsed.data;
    if (response.ok && settings.requireJsonSuccess && parsed.kind !== "json") {
      const invalidSuccess = createError({
        response,
        payload: data,
        code: invalidJson ? "invalid_json_response" : "unexpected_response",
        message: invalidJson
          ? (settings.invalidJsonMessage || "服务器返回了无法解析的 JSON 响应。")
          : (settings.unexpectedResponseMessage || "服务器返回了无法识别的响应，请刷新后重试。"),
      });
      reportDiagnostic(settings, invalidSuccess);
      throw invalidSuccess;
    }
    if (!response.ok) {
      const formattedMessage = typeof settings.messageFormatter === "function"
        ? settings.messageFormatter(data, response)
        : "";
      const error = createError({
        response,
        payload: data,
        message: formattedMessage || errorMessage(data, response),
      });
      reportDiagnostic(settings, error);
      throw error;
    }
    return data;
  }

  function createSingleFlight() {
    let locked = false;
    return Object.freeze({
      isLocked: function () {
        return locked;
      },
      tryLock: function () {
        if (locked) return false;
        locked = true;
        return true;
      },
      release: function () {
        locked = false;
      },
    });
  }

  function copyText(value) {
    const text = String(value === undefined || value === null ? "" : value);
    if (navigator.clipboard && typeof navigator.clipboard.writeText === "function") {
      return navigator.clipboard.writeText(text);
    }
    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.setAttribute("readonly", "");
    textarea.style.position = "fixed";
    textarea.style.opacity = "0";
    document.body.appendChild(textarea);
    textarea.select();
    const copied = document.execCommand("copy");
    textarea.remove();
    return copied ? Promise.resolve() : Promise.reject(new Error("浏览器不支持复制"));
  }

  function storageGet(key, fallbackValue) {
    const fallback = fallbackValue === undefined ? "" : fallbackValue;
    try {
      if (!root.localStorage) return fallback;
      const value = root.localStorage.getItem(String(key));
      return value === null ? fallback : value;
    } catch (_error) {
      return fallback;
    }
  }

  function storageSet(key, value) {
    try {
      if (!root.localStorage) return false;
      root.localStorage.setItem(String(key), String(value));
      return true;
    } catch (_error) {
      return false;
    }
  }

  function storageRemove(key) {
    try {
      if (!root.localStorage) return false;
      root.localStorage.removeItem(String(key));
      return true;
    } catch (_error) {
      return false;
    }
  }

  function storageJsonGet(key, fallbackValue) {
    const fallback = fallbackValue === undefined ? null : fallbackValue;
    const raw = storageGet(key, "");
    if (!raw) return fallback;
    try {
      return JSON.parse(raw);
    } catch (_error) {
      return fallback;
    }
  }

  function storageJsonSet(key, value) {
    try {
      return storageSet(key, JSON.stringify(value));
    } catch (_error) {
      return false;
    }
  }

  function createSubmissionId(fallbackPrefix) {
    try {
      if (root.crypto && typeof root.crypto.randomUUID === "function") {
        return root.crypto.randomUUID();
      }
    } catch (_error) {
      // Embedded browsers may expose crypto while denying access.
    }
    const bytes = new Uint32Array(4);
    try {
      if (root.crypto && typeof root.crypto.getRandomValues === "function") {
        root.crypto.getRandomValues(bytes);
      } else {
        throw new Error("secure random unavailable");
      }
    } catch (_error) {
      for (let index = 0; index < bytes.length; index += 1) {
        bytes[index] = Math.floor(Math.random() * 0xFFFFFFFF);
      }
    }
    const prefix = String(fallbackPrefix || "submission").replace(/[^A-Za-z0-9_.:-]+/g, "-");
    const random = Array.from(bytes, (value) => value.toString(36)).join("-");
    return `${prefix}-${Date.now().toString(36)}-${random}`;
  }

  root.VFIEvalShared = Object.freeze({
    copyText,
    createError,
    createSingleFlight,
    createSubmissionId,
    diagnosticIds,
    errorMessage,
    networkError,
    readResponse,
    request,
    recoverySuggestion,
    storageGet,
    storageJsonGet,
    storageJsonSet,
    storageRemove,
    storageSet,
    timeoutError,
  });
})(window);
