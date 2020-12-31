package convolution_autoencoder;


import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.primitives.Triple;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ShortGraphBuilder {


    /**
     *
     * @param feedDim
     * @param center
     */
    public static ComputationGraph grabEncoder(int feedDim, int baseK, int maxK, int center, double lr)
    {

        int layercount = 0;
        int geo = 4;
        while (feedDim > geo){
            geo = geo + geo;
            layercount++;
        }



        List<Triple<String, Layer, String>> inputlayers = inputLayers("", layercount, maxK, baseK, 0);
        List<Triple<String, Layer, String>> centerLayers = centerLayers(center);
        List<Triple<String, Layer, String>> outputLayers = outputLayers(layercount, baseK, maxK, true, 0);

        Map<String, InputPreProcessor> preprocessors = new HashMap<>();


        ComputationGraphConfiguration.GraphBuilder  confBuild = new NeuralNetConfiguration.Builder()
                .seed(123)
                .updater(new RmsProp(lr))
                .weightInit(WeightInit.XAVIER)
                .l2(1e-4)
                .graphBuilder()
                //.setInputTypes(InputType.convolutional(geo,geo, 3), InputType.feedForward(512), InputType.convolutional(1,1,512))
                .addInputs("disc_in");


        for (int i = 0; i < inputlayers.size(); i++) {
            confBuild = confBuild.addLayer(inputlayers.get(i).getFirst(), inputlayers.get(i).getSecond(), inputlayers.get(i).getThird() );
        }

        for (int i = 0; i < centerLayers.size(); i++) {
            confBuild = confBuild.addLayer(centerLayers.get(i).getFirst(), centerLayers.get(i).getSecond(), centerLayers.get(i).getThird() );
        }


        confBuild = confBuild.addLayer(outputLayers.get(0).getFirst(), outputLayers.get(0).getSecond(),
                new FeedForwardToCnnPreProcessor(4,4,baseK),outputLayers.get(0).getThird() );

        for (int i = 1; i < outputLayers.size(); i++) {
            confBuild = confBuild.addLayer(outputLayers.get(i).getFirst(), outputLayers.get(i).getSecond(), outputLayers.get(i).getThird() );
        }


        confBuild
                .setOutputs("out") // TODO: 8/18/2020 "toRGB"
                .setInputTypes(InputType.convolutional(geo,geo, 3))

                .build();

        ComputationGraph net = new ComputationGraph(confBuild.build());
        net.init();
        net.setListeners(new ScoreIterationListener(100));

        System.out.println(net.summary());

        return net;


    }


    private static List<Triple<String, Layer, String>> inputLayers(String combo, int layerCount, int maxK, int baseK, double drop)
    {

        List<Triple<String, Layer, String>> layerList = new ArrayList<>();

        int start = maxK;
        int per = (maxK-baseK)/layerCount;

        String in = "disc_in";
        if (!combo.equals("")) in = combo;
        String startName = "fromRGB";
        String layerName = "disc_";
        int lyrCnt = 1;
        layerList.add(new Triple<>(startName, new ConvolutionLayer.Builder().kernelSize(1,1).stride(1,1).nIn(3).nOut(start).activation(Activation.IDENTITY).build(), in));

        /**  Disc Layers Here **/

        for (int i = 0; i < layerCount; i++) {

            layerList.add(new Triple<>(layerName + lyrCnt, new ConvolutionLayer.Builder().kernelSize(3,3).padding(1,1).stride(1,1).nIn(start - (i*per)).nOut(start - ((i+1)*per)).dropOut(drop).activation(Activation.LEAKYRELU).build(), startName));
            startName = layerName + lyrCnt; lyrCnt++;
            //layerList.add(new Triple<>(layerName + lyrCnt, new ConvolutionLayer.Builder().kernelSize(3,3).padding(1,1).stride(1,1).nIn(512).nOut(512).dropOut(drop).activation(Activation.LEAKYRELU).build(), startName));
            //startName = layerName + lyrCnt; lyrCnt++;
            layerList.add(new Triple<>(layerName + lyrCnt, new SubsamplingLayer.Builder().kernelSize(2,2).stride(2,2).poolingType(SubsamplingLayer.PoolingType.MAX).build(), startName));
            startName = layerName + lyrCnt; lyrCnt++;
            // TODO: 8/13/2020 elim padding and subsample?
        }

        /**   -   **/

        layerList.add(new Triple<>("final_1", new ConvolutionLayer.Builder().kernelSize(3,3).padding(1,1).stride(1,1).nIn(start - ((layerCount+1)*per)).nOut(baseK).dropOut(drop).activation(Activation.LEAKYRELU).build(), startName));

        //layerList.add(new Triple<>("final_2", new ConvolutionLayer.Builder().kernelSize(4,4).padding(1,1).stride(1,1).nIn(512).nOut(512).dropOut(drop).activation(Activation.LEAKYRELU).build(), "final_1"));

        //layerList.add(new Triple<>("output", new OutputLayer.Builder(LossFunctions.LossFunction.L2).nIn(1).nOut(1).activation(Activation.SIGMOID).build(), "final_2"));
        // TODO: 8/12/2020 activation linear?
        return layerList;

    }
        //128 = 8x4x4
    private static List<Triple<String, Layer, String>> centerLayers(int centerDim)
    {
        List<Triple<String, Layer, String>> layerList = new ArrayList<>();

        layerList.add(new Triple<>("center_in", new DenseLayer.Builder().nIn(128).nOut(centerDim)
                .weightInit(WeightInit.XAVIER).activation(Activation.LEAKYRELU).build(), "final_1"));

        layerList.add(new Triple<>("center_out", new DenseLayer.Builder().nIn(centerDim).nOut(128)
                .weightInit(WeightInit.XAVIER).activation(Activation.LEAKYRELU).build(), "center_in"));

        return layerList;

    }

    private static List<Triple<String, Layer, String>> outputLayers(int layerCount, int baseK, int maxK, boolean includeLoss, double dropout)
    {

        //layerCount--;
        int start = baseK;
        int per = (maxK - baseK)/layerCount;

        List<Triple<String, Layer, String>> layerList = new ArrayList<>();
        // TODO: 8/13/2020 could switch this to padded 4x4 per writeup
        //layerList.add(new Triple<>("gen_1", new ConvolutionLayer.Builder().kernelSize(4,4).padding(3,3).stride(1,1).nIn(512).dropOut(dropout).nOut(512).activation(Activation.LEAKYRELU).build(), "center_out"));

        layerList.add(new Triple<>("gen_2", new ConvolutionLayer.Builder() .kernelSize(3,3).padding(1,1) .stride(1,1).nIn(start).dropOut(dropout).nOut(start).activation(Activation.LEAKYRELU).build(), "center_out"));
        //layerList.add(new Triple<>("gen_2", new ConvolutionLayer.Builder().kernelSize(3,3) .stride(1,1).nIn(512).nOut(512).activation(Activation.LEAKYRELU).build(), "gen_1"));



        String startName = "gen_2";
        String layerName = "gen_";
        int lyrCnt = 3;

        /**  Disc Layers Here **/

        for (int i = 0; i < layerCount; i++) {

            layerList.add(new Triple<>(layerName + lyrCnt, new Upsampling2D.Builder().size(2) .build(), startName));
            startName = layerName + lyrCnt; lyrCnt++;
            //layerList.add(new Triple<>(layerName + lyrCnt, new ConvolutionLayer.Builder().kernelSize(3,3).padding(1,1).stride(1,1).nIn(512).dropOut(dropout).nOut(512).activation(Activation.LEAKYRELU).build(), startName));
            //startName = layerName + lyrCnt; lyrCnt++;
            layerList.add(new Triple<>(layerName + lyrCnt, new ConvolutionLayer.Builder().kernelSize(3,3).padding(1,1).stride(1,1).nIn(start + (i*per)).dropOut(dropout).nOut(start + ((i+1)*per)).activation(Activation.LEAKYRELU).build(), startName));
            startName = layerName + lyrCnt; lyrCnt++;
        }

        /**   -   **/

        layerList.add(new Triple<>("toRGB", new ConvolutionLayer.Builder().kernelSize(1,1) .stride(1,1).nIn(start + (layerCount*per)).nOut(3).activation(Activation.IDENTITY).build(), startName));
        //layerList.add(new Triple<>("toRGB", new ConvolutionLayer.Builder().kernelSize(1,1) .stride(1,1).nIn(512).nOut(3).activation(Activation.IDENTITY).build(), "gen_2"));
        if (includeLoss) {
            layerList.add(new Triple<>("out", new CnnLossLayer.Builder().activation(Activation.IDENTITY).lossFunction(LossFunctions.LossFunction.MSE).build(), "toRGB"));
        }

        return layerList;


    }





}
