import java.io.File;
import java.io.Writer;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;

public class CreateFileUtil {

    public static void createJsonFile(String filePath) {
        // 生成json格式文件
        try {
            File file = new File(filePath);
            if (!file.getParentFile().exists()) {
                file.getParentFile().mkdirs();
            }
            if (file.exists()) {
                file.delete();
            }
            file.createNewFile();
        }catch (Exception e) {
                e.printStackTrace();
            }
    }

    public static boolean append(String jsonString, String filePath) {

        boolean flag = true;

        File file=new File(filePath);

        try {
            Writer write = new OutputStreamWriter(new FileOutputStream(file,true), "UTF-8");
            write.write(jsonString);
            write.flush();
            write.close();
        } catch (Exception e) {
            flag = false;
            e.printStackTrace();
        }
        return flag;
    }
}

