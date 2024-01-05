package uk.co.workingedge.gemini.x

import com.google.ai.client.generativeai.GenerativeModel
import com.google.ai.client.generativeai.type.SafetySetting
import com.google.ai.client.generativeai.type.generationConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class GeminiX {
    companion object {
        var generativeModel: GenerativeModel? = null;

        fun init(
            modelName: String,
            apiKey: String,
            config: Map<String, Any>?,
            safetySettings: List<SafetySetting>?,
        ) {
            val thisConfig = generationConfig {
                config?.forEach { (key, value) ->
                    when (key) {
                        "temperature" -> temperature = value as Float
                        "topK" -> topK = value as Int
                        "topP" -> topP = value as Float
                        "maxOutputTokens" -> maxOutputTokens = value as Int
                        "stopSequences" -> stopSequences = value as List<String>
                    }
                }
            }
            generativeModel = GenerativeModel(modelName, apiKey, thisConfig, safetySettings)
        }

        fun sendMessage(
            inputText: String,
            successCallback: (String) -> Unit,
            errorCallback: (String) -> Unit,
        ) {
            // TODO support images for multi-modal models

            GlobalScope.launch(Dispatchers.IO) {
                val response = try {
                    generativeModel?.generateContent(inputText)
                } catch (e: Exception) {
                    e.message?.let { errorCallback(it) }
                    null
                }

                withContext(Dispatchers.Main) {
                    if (response != null) {
                        response.text?.let { successCallback(it) }
                    }
                }
            }
        }

        fun countTokens(
            inputText: String,
            successCallback: (Int) -> Unit,
            errorCallback: (String) -> Unit,
        ) {
            // TODO support images for multi-modal models

            GlobalScope.launch(Dispatchers.IO) {
                val response = try {
                    generativeModel?.countTokens(inputText)
                } catch (e: Exception) {
                    e.message?.let { errorCallback(it) }
                    null
                }

                withContext(Dispatchers.Main) {
                    if (response != null) {
                        response.totalTokens?.let { successCallback(it) }
                    }
                }
            }
        }
    }
}
