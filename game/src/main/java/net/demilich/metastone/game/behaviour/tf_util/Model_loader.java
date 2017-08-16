package net.demilich.metastone.game.behaviour.tf_util;

import net.demilich.metastone.game.behaviour.GreedyOptimizeMoveLinear;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tensorflow.DataType;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class Model_loader {
    public Graph g = new Graph();
    public Session s;
    private final static Logger logger = LoggerFactory.getLogger(GreedyOptimizeMoveLinear.class);

    public Model_loader(String model_dir, String model_name) {
        logger.info("Loading model");
        try{
            byte[] graphDef = Files.readAllBytes(Paths.get(model_dir, model_name));
            g.importGraphDef(graphDef);
            this.s = new Session(g);
        }
        catch (IOException exception){
            exception.printStackTrace();
        }
    }

}
