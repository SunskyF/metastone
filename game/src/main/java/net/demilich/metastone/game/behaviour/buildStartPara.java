package net.demilich.metastone.game.behaviour;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;

public class buildStartPara {
    private final static Logger logger = LoggerFactory.getLogger(buildStartPara.class);

    public static void main(String[] args) {
        INDArray para;
        try{
            logger.info("Loading...");
            File readFile = new File("app/NdModel/linear/98feaGameTree_mean_para_ES_4zero.data");
            para = Nd4j.readBinary(readFile);
            INDArray w = para.get(NDArrayIndex.interval(0,96), NDArrayIndex.all());
            INDArray b = para.get(NDArrayIndex.point(96), NDArrayIndex.all());
            logger.info("Para: {}", para);
            logger.info("w: {}", w);
            logger.info("b: {}", b);
            logger.info("{} {}", w.shape(), b.shape());
            logger.info("After Shape: {}", para.shape());
            para = Nd4j.concat(0, w, Nd4j.randn(2, 1), b);
            logger.info("Para: {}", para);
            logger.info("Shape: {}", para.shape());
//            File saveFile = new File("app/NdModel/linear/startPoint_98feature.data");
//            Nd4j.saveBinary(para, saveFile);
        }
        catch (IOException e){
            e.printStackTrace();
        }
    }
}
