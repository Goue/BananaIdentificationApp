package com.example.myapplication;

import androidx.annotation.Nullable;
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Typeface;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.example.myapplication.ml.ModelUnquant;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MainActivity extends AppCompatActivity {
    //宣告
    TextView result, classified;
    ImageView imageView;
    Button picture,bnt;
    int imageSize = 224;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        //宣告
        result = findViewById(R.id.result);
        classified=findViewById(R.id.classified);
        imageView = findViewById(R.id.imageView);
        picture = findViewById(R.id.button);
        bnt=findViewById(R.id.button2);

        //當按下Take Picture時
        picture.setOnClickListener(new View.OnClickListener() {
            @RequiresApi(api = Build.VERSION_CODES.M)
            @Override
            public void onClick(View view) {
                // 如果有權限開啟相機
                if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(cameraIntent, 1);
                } else {
                    //沒權限則請求權限
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
                }
            }
        });

        //當按下Choose Picture時
        bnt.setOnClickListener(new View.OnClickListener() {
            @RequiresApi(api = Build.VERSION_CODES.M)
            @Override
            public void onClick(View view) {
                //如果有權限開啟圖庫
                if (checkSelfPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED) {
                Intent intent = new Intent(
                        Intent.ACTION_PICK,
                        android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(intent, 2);
                }else{//沒權限則請求權限
                    requestPermissions(new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}, 100);
                }
            }
        });

    }
    //返回照相獲選照片結果
    @Override
    public void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        //判斷是否為照相
        if (requestCode == 1 && resultCode == RESULT_OK) {
            //取得照片
            Bitmap image = (Bitmap) data.getExtras().get("data");
            //調整圖片大小
            int dimension = Math.min(image.getWidth(), image.getHeight());
            image = ThumbnailUtils.extractThumbnail(image, dimension, dimension);
            //顯示在imageView
            imageView.setImageBitmap(image);
            //設定image的大小適合機器學習的模型並呼叫
            image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
            classifyImage(image);
        }
        //判斷是否為從圖庫選照片
        if (requestCode == 2 && resultCode == RESULT_OK) {
            try {
                //獲取系統返回的照片的Uri
                Uri selectedImage = data.getData();
                String[] filePathColumn = {MediaStore.Images.Media.DATA};
                Cursor cursor = getContentResolver().query(selectedImage, filePathColumn, null, null, null);//從系統表中查詢指定Uri對應的照片
                cursor.moveToFirst();
                int columnIndex = cursor.getColumnIndex(filePathColumn[0]);
                //獲取照片路徑
                String path = cursor.getString(columnIndex);
                Bitmap image = BitmapFactory.decodeFile(path);
                //顯示在imageView
                imageView.setImageBitmap(image);
                //設定image的大小適合機器學習的模型並呼叫
                image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
                classifyImage(image);
                cursor.close();
            } catch (Exception e) {
                // TODO Auto-generatedcatch block
                e.printStackTrace();
            }
            super.onActivityResult(requestCode, resultCode, data);
        }
    }

    //機器學習
    //模型匯入從File>New>Other>TensorFlow Lite Model，並從資料夾中選取Model資料夾中的model_unquant
    public void classifyImage(Bitmap image){
        try {
            ModelUnquant model = ModelUnquant.newInstance(getApplicationContext());

            //創建輸入以供參考
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4*imageSize*imageSize*3);
            byteBuffer.order(ByteOrder.nativeOrder());

            int [] intValues=new int[imageSize*imageSize];
            image.getPixels(intValues,0,image.getWidth(),0,0,image.getWidth(),image.getHeight());
            int pixel=0;
            for(int i=0;i<imageSize;i++){
                for(int j=0;j<imageSize;j++){
                    int val=intValues[pixel++];//RGB
                    byteBuffer.putFloat(((val >> 16)&0xFF)*(1.f/255.f));
                    byteBuffer.putFloat(((val >> 8)&0xFF)*(1.f/255.f));
                    byteBuffer.putFloat((val & 0xFF)*(1.f/255.f));
                }
            }

            inputFeature0.loadBuffer(byteBuffer);

            //運行模型並獲得結果
            ModelUnquant.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            //取得機率最大的結果
            float[] confidences = outputFeature0.getFloatArray();
            int maxPos = 0;
            float maxConfidence = 0;
            for(int i =0;i<confidences.length;i++){
                if(confidences[i]>maxConfidence){
                    maxConfidence=confidences[i];
                    maxPos=i;
                }
            }
            //顯示結果
            String[] classes={"未熟\n" +
                    "能穩定血糖，增加飽足感幫助瘦身","未熟\n" +
                    "能穩定血糖，增加飽足感幫助瘦身","熟成\n" +
                    "有助代謝，消除便秘","過熟\n" +
                    "幫助抗癌，可助延緩老化"};
            result.setText(classes[maxPos]);

            // 如果不再使用，則釋放模型資源
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }

    }//使用模擬器版本: Pixel 4 API30

}

