package com.retinavit

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import org.pytorch.LiteModuleLoader
import org.pytorch.Module
import java.io.File
import java.io.FileOutputStream

class MainActivity : AppCompatActivity() {
    private var model: Module? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Load TorchScript model from assets
        val modelPath = assetFilePath("retinavit_mobile.pt")
        model = LiteModuleLoader.load(modelPath)
        
        // Placeholder for on-device inference logic
    }

    private fun assetFilePath(assetName: String): String {
        val file = File(filesDir, assetName)
        if (file.exists() && file.length() > 0) {
            return file.absolutePath
        }
        assets.open(assetName).use { inputStream ->
            FileOutputStream(file).use { outputStream ->
                val buffer = ByteArray(4 * 1024)
                var read: Int
                while (inputStream.read(buffer).also { read = it } != -1) {
                    outputStream.write(buffer, 0, read)
                }
                outputStream.flush()
            }
        }
        return file.absolutePath
    }
}
