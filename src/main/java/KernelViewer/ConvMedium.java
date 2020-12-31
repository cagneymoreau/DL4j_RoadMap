package KernelViewer;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;

/**
 *
 *  Here Im using a pretrained cifar dataset
 *
 *
 */



public class ConvMedium {

    MultiLayerNetwork network;

    public static void main(String[] args)
    {
        ConvMedium c = new ConvMedium();
    }


    public ConvMedium()
    {
        try {
            network = MultiLayerNetwork.load(new File("D:\\Dropbox\\Apps\\models\\model.dl4j"), true);
            System.out.print(network.summary());
        }catch (Exception e){
            e.printStackTrace();
        }

        CovOutViewer covOutViewer = new CovOutViewer(network);

        INDArray target = Nd4j.zeros(1,10);
        target.putScalar(3L, 1);

        covOutViewer.networkReverse(3,32,32,true, target);


    }





}
