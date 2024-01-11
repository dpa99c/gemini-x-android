@file:OptIn(DelicateCoroutinesApi::class)

package uk.co.workingedge.gemini.x.lib

import android.graphics.Bitmap
import com.google.ai.client.generativeai.Chat
import com.google.ai.client.generativeai.GenerativeModel
import com.google.ai.client.generativeai.type.BlobPart
import com.google.ai.client.generativeai.type.BlockThreshold
import com.google.ai.client.generativeai.type.Content
import com.google.ai.client.generativeai.type.HarmCategory
import com.google.ai.client.generativeai.type.ImagePart
import com.google.ai.client.generativeai.type.Part
import com.google.ai.client.generativeai.type.SafetySetting
import com.google.ai.client.generativeai.type.TextPart
import com.google.ai.client.generativeai.type.content
import com.google.ai.client.generativeai.type.generationConfig
import kotlinx.coroutines.DelicateCoroutinesApi
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.flow.onCompletion
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

interface HistoryPart{
    val type: String
    val content: Any
}

class TextHistoryPart(override val content: String): HistoryPart {
    override val type: String
        get() = "text"
}

class ImageHistoryPart(override val content: Bitmap): HistoryPart {
    override val type: String
        get() = "image"
}

class BlobHistoryPart(override val content: ByteArray, val mimeType: String): HistoryPart {
    override val type: String
        get() = "blob"
}

class HistoryItem (val parts: List<HistoryPart>, val isUser: Boolean)

class GeminiX {
    companion object {
        private var generativeModel: GenerativeModel? = null
        private var chat: Chat? = null

        fun init(
            modelName: String,
            apiKey: String,
            config: Map<String, Any>?,
            safetySettings: Map<String, String>?,
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
            val thisSafetySettings: List<SafetySetting> = listOf()
            if(safetySettings != null) {
                safetySettings.forEach { (key, value) ->
                    when (key) {
                        "HARASSMENT" -> thisSafetySettings.plus(SafetySetting(HarmCategory.HARASSMENT, getHarmLevel(value)))
                        "HATE_SPEECH" -> thisSafetySettings.plus(SafetySetting(HarmCategory.HATE_SPEECH, getHarmLevel(value)))
                        "SEXUALLY_EXPLICIT" -> thisSafetySettings.plus(SafetySetting(HarmCategory.SEXUALLY_EXPLICIT, getHarmLevel(value)))
                        "DANGEROUS_CONTENT" -> thisSafetySettings.plus(SafetySetting(HarmCategory.DANGEROUS_CONTENT, getHarmLevel(value)))
                        "UNSPECIFIED" -> thisSafetySettings.plus(SafetySetting(HarmCategory.UNKNOWN, getHarmLevel(value)))
                    }
                }
            }

            generativeModel = GenerativeModel(modelName, apiKey, thisConfig, thisSafetySettings)
        }

        fun sendMessage(
            successCallback: (String, Boolean) -> Unit,
            errorCallback: (String) -> Unit,
            inputText: String,
            images: List<Bitmap>? = null,
            streamResponse: Boolean? = false,
        ) {
            if(generativeModel == null) {
                errorCallback("Model not initialized")
                return
            }
            val inputContent = content {
                role = "user"
                for (image in images ?: listOf()) {
                    image(image)
                }
                text(inputText)
            }

            GlobalScope.launch(Dispatchers.IO) {
                if(streamResponse == true){
                    try {
                        generativeModel?.generateContentStream(inputContent)?.onCompletion {
                            withContext(Dispatchers.Main) {
                                successCallback("", true)
                            }
                        }?.collect { chunk ->
                            withContext(Dispatchers.Main) {
                                chunk.text?.let { successCallback(it, false) }
                            }
                        }
                    } catch (e: Exception) {
                        e.message?.let { errorCallback(it) }
                    }

                }else{
                    val response = try {
                        generativeModel?.generateContent(inputContent)
                    } catch (e: Exception) {
                        e.message?.let { errorCallback(it) }
                        null
                    }

                    withContext(Dispatchers.Main) {
                        if (response != null) {
                            response.text?.let { successCallback(it, false) }
                        }
                    }
                }
            }
        }

        fun countTokens(
            successCallback: (Int) -> Unit,
            errorCallback: (String) -> Unit,
            inputText: String,
            images: List<Bitmap>? = null,
        ) {
            if(generativeModel == null) {
                errorCallback("Model not initialized")
                return
            }

            val inputContent = content {
                role = "user"
                for (image in images ?: listOf()) {
                    image(image)
                }
                text(inputText)
            }

            GlobalScope.launch(Dispatchers.IO) {
                val response = try {
                    generativeModel?.countTokens(inputContent)
                } catch (e: Exception) {
                    e.message?.let { errorCallback(it) }
                    null
                }

                withContext(Dispatchers.Main) {
                    response?.totalTokens?.let { successCallback(it) }
                }
            }
        }

        fun initChat(
            successCallback: () -> Unit,
            errorCallback: (String) -> Unit,
            history: List<HistoryItem>? = listOf(),
        ) {
            if(generativeModel == null) {
                errorCallback("Model not initialized")
                return
            }
            GlobalScope.launch(Dispatchers.IO) {
                try {
                    var chatHistory:List<Content> = listOf()
                    history?.forEach { item ->
                        val role = if (item.isUser) "user" else "model"
                        var parts:List<Part> = mutableListOf()
                        item.parts.forEach { part ->
                            val contentPart:Part
                            when (part.type) {
                                "text" -> {
                                    contentPart = TextPart(part.content as String)
                                }
                                "image" -> {
                                    contentPart = ImagePart(part.content as Bitmap)
                                }
                                "blob" -> {
                                    val blobPart = part.content as BlobHistoryPart
                                    contentPart = BlobPart(blobPart.mimeType, blobPart.content)
                                }
                                else -> {
                                    throw Exception("Unknown history part type: ${part.type}")
                                }
                            }
                            parts = parts.plus(contentPart)
                        }
                        val content = Content(role, parts)
                        chatHistory = chatHistory.plus(content)
                    }
                    chat = generativeModel?.startChat(chatHistory)
                } catch (e: Exception) {
                    e.message?.let { errorCallback(it) }
                }

                withContext(Dispatchers.Main) {
                    successCallback()
                }
            }
        }

        fun sendChatMessage(
            successCallback: (String, Boolean) -> Unit,
            errorCallback: (String) -> Unit,
            inputText: String,
            images: List<Bitmap>? = null,
            streamResponse: Boolean? = false,
        ) {
            if(generativeModel == null) {
                errorCallback("Model not initialized")
                return
            }
            if(chat == null) {
                errorCallback("Chat not initialized")
                return
            }

            val inputContent = content {
                role = "user"
                for (image in images ?: listOf()) {
                    image(image)
                }
                text(inputText)
            }

            GlobalScope.launch(Dispatchers.IO) {
                if(streamResponse == true){
                    try {
                        chat?.sendMessageStream(inputContent)?.onCompletion {
                            withContext(Dispatchers.Main) {
                                successCallback("", true)
                            }
                        }?.collect { chunk ->
                            withContext(Dispatchers.Main) {
                                chunk.text?.let { successCallback(it, false) }
                            }
                        }
                    } catch (e: Exception) {
                        e.message?.let { errorCallback(it) }
                    }
                }else{
                    val response = try {
                        chat?.sendMessage(inputContent)
                    } catch (e: Exception) {
                        e.message?.let { errorCallback(it) }
                        null
                    }

                    withContext(Dispatchers.Main) {
                        if (response != null) {
                            response.text?.let { successCallback(it, true) }
                        }
                    }
                }
            }
        }

        fun countChatTokens(
            successCallback: (Int) -> Unit,
            errorCallback: (String) -> Unit,
            inputText: String? = null,
            images: List<Bitmap>? = null,
        ) {
            if(generativeModel == null) {
                errorCallback("Model not initialized")
                return
            }
            if(chat == null) {
                errorCallback("Chat not initialized")
                return
            }

            GlobalScope.launch(Dispatchers.IO) {
                val response = try {
                    val history = chat?.history
                    for (image in images ?: listOf()) {
                        history?.add(Content("user", listOf(ImagePart(image))))
                    }
                    if (inputText != null) {
                        history?.add(Content("user", listOf(TextPart(inputText))))
                    }
                    history?.let { generativeModel?.countTokens(*it.toTypedArray()) }
                } catch (e: Exception) {
                    e.message?.let { errorCallback(it) }
                    null
                }

                withContext(Dispatchers.Main) {
                    response?.totalTokens?.let { successCallback(it) }
                }
            }
        }

        fun getChatHistory(
            successCallback: (List<HistoryItem>) -> Unit,
            errorCallback: (String) -> Unit,
        ) {
            if(generativeModel == null) {
                errorCallback("Model not initialized")
                return
            }
            if(chat == null) {
                errorCallback("Chat not initialized")
                return
            }

            val history = chat?.history
            var historyItems:List<HistoryItem> = listOf()
            history?.forEach { content ->
                val isUser = content.role == "user"
                var parts:List<HistoryPart> = listOf()
                content.parts.forEach { part ->
                    val historyPart:HistoryPart
                    when (part) {
                        is TextPart -> {
                            historyPart = TextHistoryPart(part.text)
                        }
                        is ImagePart -> {
                            historyPart = ImageHistoryPart(part.image)
                        }
                        is BlobPart -> {
                            historyPart = BlobHistoryPart(part.blob, part.mimeType)
                        }
                        else -> {
                            throw Exception("Unknown history part type: ${part::class.java}")
                        }
                    }
                    parts = parts.plus(historyPart)
                }
                historyItems = historyItems.plus(HistoryItem(parts, isUser))
            }
            successCallback(historyItems)
        }


        /********************
         * Internal functions
         *******************/
        private fun getHarmLevel(level:String) : BlockThreshold {
            return when (level) {
                "NONE" -> BlockThreshold.NONE
                "ONLY_HIGH" -> BlockThreshold.ONLY_HIGH
                "MEDIUM_AND_ABOVE" -> BlockThreshold.MEDIUM_AND_ABOVE
                "LOW_AND_ABOVE" -> BlockThreshold.LOW_AND_ABOVE
                else -> BlockThreshold.UNSPECIFIED
            }
        }
    }
}
