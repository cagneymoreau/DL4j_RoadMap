package KernelViewer;


import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.loader.ImageLoader;
import org.datavec.image.loader.Java2DNativeImageLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.BaseImageRecordReader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.zoo.model.YOLO2;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.List;
import java.util.Map;

/**
 * This is the yolo network
 *
 *
 */

public class ConvAdvanced {



    public static void main(String[] args)
    {
        ConvAdvanced convAdvanced = new ConvAdvanced();
        String image = "D:\\Dropbox\\cagney\\brandon\\2017-10-08 17.26.52.jpg";
        ImagePreProcessingScaler preProcessingScaler = new ImagePreProcessingScaler(0,1);

        try {

            NativeImageLoader loader = new NativeImageLoader(416, 416, 3);
            //ImageLoader imageLoader = new ImageLoader(416, 416, 3);

            INDArray input = loader.asMatrix(new File(image)); // imageLoader.toRaveledTensor (new File(image));

            preProcessingScaler.transform(input);

            convAdvanced.preTrianedNet(input);

        }catch (Exception e)
        {
            e.printStackTrace();
        }

    }


    public ConvAdvanced()
    {

    }


    public void preTrianedNet(INDArray input) throws Exception
    {

        ComputationGraph yolo = (ComputationGraph) YOLO2.builder().build().initPretrained();

        System.out.print(yolo.summary());
        Layer[] names = yolo.getLayers();
        CovOutViewer cv = new CovOutViewer(yolo);

        //INDArray[] out = yolo.output(input);
        Map<String, INDArray> a = yolo.feedForward(new INDArray[] {input}, names.length-1, false);
        //Yolo2OutputLayer outputLayer = (Yolo2OutputLayer) yolo.getOutputLayer(0);
        //List<DetectedObject> predictions = outputLayer.getPredictedObjects(out[0], .45f);

        cv.update();


        //System.out.print("Done");

    }



}
