package com.company;

import weka.classifiers.Classifier;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Instances;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;

import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;

public class Main {

    public static void main(String[] args) {
        DataSource source;
        Instances testData;

        try
        {
            System.out.println("Start");

            // On crée le fichier de sortie à partir du fichier mis en argument
            PrintWriter writer = new PrintWriter(args[1], "UTF-8");

            // On charge les données à partir du fichier mis en argument
            source = new DataSource(args[0]);
            testData = source.getDataSet();

            // On enleve le TrackID et le SampleID pour permettre la cassification
            testData.deleteAttributeAt(0);
            testData.deleteAttributeAt(0);
            testData.setClassIndex(testData.numAttributes() - 1);

            // On normalize les données d'entrée
            Normalize normalization = new Normalize();
            normalization.setInputFormat(testData);
            Instances normalizedData = Filter.useFilter(testData, normalization);

            // On charge le model déjà entrainé
            Classifier cls = (Classifier) weka.core.SerializationHelper.read("libSVM-ssd.model");

            // On crée un array des classes existantess afin d'écrire un document de sortie lisible
            Enumeration<Object> enumeratedClasses = normalizedData.classAttribute().enumerateValues();
            ArrayList<Object> classes = Collections.list(enumeratedClasses);
            Object[] classArray = classes.toArray();

            // On classifie
            System.out.println("Classify");
            for (int i = 0; i < normalizedData.numInstances(); i++)
            {
                // Identification de la track numérique (le nombre correspond à la classe)
                double trackClass = cls.classifyInstance(normalizedData.instance(i));
                // Identification de la track nominal
                String classification= "" + classArray[(int)trackClass].toString();
                // On écrit
                writer.println(classification);
            }
            writer.close();
            System.out.println("Finished");

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
