/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package deathevent;

import java.io.File;
import java.io.IOException;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ArffLoader;

/**
 *
 * @author daothikieutrang
 */
public class NaiveBayesModel {

//    huấn luyện mô hình 
    public NaiveBayes train() throws IOException, Exception {
        ArffLoader trainLoader = new ArffLoader();
        trainLoader.setFile(new File("F:\\KPDL\\btl_naviebayes\\KPDL_DeathEvent_NaiveBayes\\DeathEvent\\src\\deathevent\\data_train.arff"));
        Instances trainData = trainLoader.getDataSet();
        trainData.setClassIndex(trainData.numAttributes() - 1);

        NaiveBayes nb = new NaiveBayes();
        nb.buildClassifier(trainData);
        return nb;
    }

//    đánh giá mô hình
    public void test() throws Exception {
        NaiveBayes nb = train();

        ArffLoader testLoader = new ArffLoader();
        testLoader.setFile(new File("F:\\KPDL\\btl_naviebayes\\KPDL_DeathEvent_NaiveBayes\\DeathEvent\\src\\deathevent\\data_test.arff"));
        Instances testData = testLoader.getDataSet();
        testData.setClassIndex(testData.numAttributes() - 1);

        int correctClassifications = 0;
        int incorrectClassifications = 0;

        for (int i = 0; i < testData.numInstances(); i++) {
            
            double actualClass = testData.instance(i).classValue();
            String actualClassName = testData.classAttribute().value((int) actualClass);
            
            Instance newInst = testData.instance(i);
            double predClass = nb.classifyInstance(newInst);
            String predClassName = testData.classAttribute().value((int) predClass);
            
            if (predClassName.equals(actualClassName)) {
                correctClassifications++;
            } else {
                incorrectClassifications++;
            }

        }
        int totalTestInstances = testData.numInstances();
        double accuracy = (double) correctClassifications / totalTestInstances;
        double errorRate = (double) incorrectClassifications / totalTestInstances;
        System.out.println("Correctly Classified Instances  " + correctClassifications + " " + (accuracy * 100) + " %");
        System.out.println("Incorrectly Classified Instances " + incorrectClassifications + " " + (errorRate * 100) + " %");
    }

//   dự đoán 1 mẫu mới
    public String predict(String age, String anaemia, String creatinine_phosphokinase, String diabetes, String ejection_fraction, String high_blood_pressure, String platelets, String serum_creatinine, String serum_sodium, String sex, String smoking, String time) throws IOException, Exception {
        NaiveBayes nb = train();

        ArffLoader testLoader = new ArffLoader();
        testLoader.setFile(new File("F:\\KPDL\\btl_naviebayes\\KPDL_DeathEvent_NaiveBayes\\DeathEvent\\src\\deathevent\\data_test.arff"));
        Instances testData = testLoader.getDataSet();
        testData.setClassIndex(testData.numAttributes() - 1);

        Instance newInstance = new DenseInstance(testData.numAttributes());
        newInstance.setDataset(testData);

        newInstance.setValue(testData.attribute("age"), age);
        newInstance.setValue(testData.attribute("anaemia"), anaemia);
        newInstance.setValue(testData.attribute("creatinine_phosphokinase"), creatinine_phosphokinase);
        newInstance.setValue(testData.attribute("diabetes"), diabetes);
        newInstance.setValue(testData.attribute("ejection_fraction"), ejection_fraction);
        newInstance.setValue(testData.attribute("high_blood_pressure"), high_blood_pressure);
        newInstance.setValue(testData.attribute("platelets"), platelets);
        newInstance.setValue(testData.attribute("serum_creatinine"), serum_creatinine);
        newInstance.setValue(testData.attribute("serum_sodium"), serum_sodium);
        newInstance.setValue(testData.attribute("sex"), sex);
        newInstance.setValue(testData.attribute("smoking"), smoking);
        newInstance.setValue(testData.attribute("time"), time);

        double predictedClass = nb.classifyInstance(newInstance);
        String predictedClassName = testData.classAttribute().value((int) predictedClass);
        String result = "";
        if (predictedClassName.equals("0")) {
            result = "Alive";
        } else {
            result = "Dead";
        }

        return result;
    }

    public static void main(String[] args) throws Exception {
        NaiveBayesModel naive = new NaiveBayesModel();
        naive.test();
        String result= naive.predict("0_48", "0", "0_894", "1", "26_36","0", "0_116756", "0_1.8","0_119", "0","1","36_66");
        System.out.println("Predict:" + result);
    }
}
