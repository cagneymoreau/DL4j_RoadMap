package KernelViewer;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.Upsampling2D;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.RmsProp;

public class ResizeNets {




    public static ComputationGraph buildEnlargeNet(int grow,int depth, int width, int height)
    {
        ComputationGraphConfiguration.GraphBuilder  confBuild = new NeuralNetConfiguration.Builder()
                .seed(123)
                .updater(new RmsProp(1e-3))
                .weightInit(WeightInit.XAVIER)
                .l2(1e-4)
                .graphBuilder()
                .addInputs("input");
        String input = "input";
        String layerName;
        if (grow == 0) {

            confBuild = confBuild.addLayer("up_1", new DenseLayer.Builder().nIn(width * height).nOut(width * height)
                    .activation(Activation.IDENTITY).build(), input);

        }else {
            for (int i = 0; i < grow; i++) {
                layerName = "up_" + (i + 1);
                confBuild = confBuild.addLayer(layerName, new Upsampling2D.Builder().size(2).build(), input);
                input = layerName;
            }
        }

        confBuild
                .setOutputs(input)
                .setInputTypes(InputType.convolutional(height, width, depth))
                .build();


        ComputationGraph net = new ComputationGraph(confBuild.build());
        net.init();
        //net.setListeners(new ScoreIterationListener(100));
        //net.gradient()
        return net;

    }



    public static ComputationGraph buildReduceNet(int shrink, int depth, int width, int height)
    {
        ComputationGraphConfiguration.GraphBuilder  confBuild = new NeuralNetConfiguration.Builder()
                .seed(123)
                .updater(new RmsProp(1e-3))
                .weightInit(WeightInit.XAVIER)
                .l2(1e-4)
                .graphBuilder()
                .addInputs("input");
        String input = "input";
        String layerName;
        if (shrink == 0) {

            confBuild = confBuild.addLayer("sub_1", new DenseLayer.Builder().nIn(width * height).nOut(width * height)
                    .activation(Activation.IDENTITY).build(), input);

        }else {
            for (int i = 0; i < shrink; i++) {
                layerName = "sub_" + (i + 1);
                confBuild = confBuild.addLayer(layerName, new SubsamplingLayer.Builder().kernelSize(2, 2).stride(2, 2).poolingType(SubsamplingLayer.PoolingType.AVG).build(), input);
                input = layerName;
            }
        }

        confBuild
                .setOutputs(input)
                .setInputTypes(InputType.convolutional(height, width, depth))
                .build();


        ComputationGraph net = new ComputationGraph(confBuild.build());
        net.init();
        //net.setListeners(new ScoreIterationListener(100));
        //net.gradient()
        return net;

    }


}
