@file:OptIn(DelicateCoroutinesApi::class)

package uk.co.workingedge.gemini.x.lib

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Bitmap.CompressFormat
import android.net.Uri
import android.provider.MediaStore
import android.util.Base64
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
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream

interface Image{
    val data: Bitmap
    val mimeType: String? get() = null
}

interface HistoryPart{
    val type: String
    val content: Any
}

class TextHistoryPart(override val content: String): HistoryPart {
    override val type: String
        get() = "text"
}

class ImageHistoryPart(override val content: Bitmap, val mimeType: String? = null): HistoryPart {
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

        /***********************************************************************
         * Gemini SDK functions
         **********************************************************************/
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
            safetySettings?.forEach { (key, value) ->
                when (key) {
                    "HARASSMENT" -> thisSafetySettings.plus(SafetySetting(HarmCategory.HARASSMENT, getHarmLevel(value)))
                    "HATE_SPEECH" -> thisSafetySettings.plus(SafetySetting(HarmCategory.HATE_SPEECH, getHarmLevel(value)))
                    "SEXUALLY_EXPLICIT" -> thisSafetySettings.plus(SafetySetting(HarmCategory.SEXUALLY_EXPLICIT, getHarmLevel(value)))
                    "DANGEROUS_CONTENT" -> thisSafetySettings.plus(SafetySetting(HarmCategory.DANGEROUS_CONTENT, getHarmLevel(value)))
                    "UNSPECIFIED" -> thisSafetySettings.plus(SafetySetting(HarmCategory.UNKNOWN, getHarmLevel(value)))
                }
            }

            generativeModel = GenerativeModel(modelName, apiKey, thisConfig, thisSafetySettings)
        }

        fun sendMessage(
            successCallback: (String, Boolean) -> Unit,
            errorCallback: (String) -> Unit,
            inputText: String,
            images: List<Image>? = null,
            streamResponse: Boolean? = false,
        ) {
            if(generativeModel == null) {
                errorCallback("Model not initialized")
                return
            }
            val inputContent = buildUserInputContent(inputText, images)

            GlobalScope.launch(Dispatchers.IO) {
                if(streamResponse == true){
                    try {
                        var fullResponse = ""
                        generativeModel?.generateContentStream(inputContent)?.onCompletion {
                            withContext(Dispatchers.Main) {
                                if(fullResponse != ""){
                                    successCallback(fullResponse, true)
                                }
                            }
                        }?.collect { chunk ->
                            withContext(Dispatchers.Main) {
                                chunk.text?.let {
                                    fullResponse += it
                                    successCallback(it, false)
                                }
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
                            response.text?.let { successCallback(it, true) }
                        }
                    }
                }
            }
        }

        fun countTokens(
            successCallback: (Int) -> Unit,
            errorCallback: (String) -> Unit,
            inputText: String,
            images: List<Image>? = null,
        ) {
            if(generativeModel == null) {
                errorCallback("Model not initialized")
                return
            }

            val inputContent = buildUserInputContent(inputText, images)

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
                            val contentPart:Part = when (part.type) {
                                "text" -> {
                                    TextPart(part.content as String)
                                }

                                "image" -> {
                                    ImagePart(part.content as Bitmap)
                                }

                                "blob" -> {
                                    val blobPart = part as BlobHistoryPart
                                    BlobPart(blobPart.mimeType, blobPart.content)
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
            images: List<Image>? = null,
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

            val inputContent = buildUserInputContent(inputText, images)

            GlobalScope.launch(Dispatchers.IO) {
                if(streamResponse == true){
                    try {
                        var finalResponse = ""
                        chat?.sendMessageStream(inputContent)?.onCompletion {
                            withContext(Dispatchers.Main) {
                                if(finalResponse != ""){
                                    successCallback(finalResponse, true)
                                }
                            }
                        }?.collect { chunk ->
                            withContext(Dispatchers.Main) {
                                chunk.text?.let {
                                    finalResponse += it
                                    successCallback(it, false)
                                }
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
            images: List<Image>? = null,
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
                    val userInput = buildUserInputContent(inputText, images)
                    val fullHistory = history?.plus(userInput)
                    fullHistory?.let { generativeModel?.countTokens(*it.toTypedArray()) }
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
                    val historyPart:HistoryPart = when (part) {
                        is TextPart -> {
                            TextHistoryPart(part.text)
                        }

                        is ImagePart -> {
                            ImageHistoryPart(part.image)
                        }

                        is BlobPart -> {
                            BlobHistoryPart(part.blob, part.mimeType)
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

        /***********************************************************************
         * Helper functions
         **********************************************************************/
        fun getModelImage(image: JSONObject, context: Context): Image {
            val bitmap = getBitmapFromUri(image.getString("uri"), context)
            if(image.has("mimeType")) {
                val mimeType = image.getString("mimeType")
                return object : Image {
                    override val data: Bitmap
                        get() = bitmap
                    override val mimeType: String
                        get() = mimeType
                }
            }else{
                return object : Image {
                    override val data: Bitmap
                        get() = bitmap
                }
            }
        }

        fun getModelImages(images: JSONArray, context: Context): List<Image> {
            val modelImages = mutableListOf<Image>()
            for (i in 0 until images.length()) {
                val image = getModelImage(images.getJSONObject(i), context)
                modelImages.add(image)
            }
            return modelImages
        }

        fun getBitmapFromUri(uri: String, context: Context): Bitmap {
            return MediaStore.Images.Media.getBitmap(
                context.contentResolver,
                Uri.parse(uri)
            )
        }

        fun bitmapToWebpByteArray(bitmap: Bitmap): ByteArray {
            val baos = ByteArrayOutputStream()

            var compressFormat: CompressFormat
            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.R) {
                compressFormat = CompressFormat.WEBP_LOSSLESS;
            }else{
                @Suppress("DEPRECATION")
                compressFormat = CompressFormat.WEBP;
            }
            bitmap.compress(compressFormat, 100, baos)
            return baos.toByteArray()
        }

        fun bitmapToBase64(bitmap: Bitmap): String {
            val bytes = bitmapToWebpByteArray(bitmap)
            return Base64.encodeToString(bytes, Base64.DEFAULT)
        }

        fun buildUserInputContent(text: String? = null, images: List<Image>? = null): Content {
            return content {
                role = "user"
                for (image in images ?: listOf()) {
                    if(image.mimeType != null){
                        val bytes = bitmapToWebpByteArray(image.data)
                        blob(image.mimeType!!, bytes)
                    }else{
                        image(image.data)
                    }
                }
                if (text != null){
                    text(text)
                }
            }
        }

        fun bitmapToByteArray(bitmap: Bitmap): ByteArray {
            val baos = ByteArrayOutputStream()
            bitmap.compress(CompressFormat.WEBP, 100, baos)
            return baos.toByteArray()
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
