package ml;

import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class Bayes {

    public static List<Integer> classify(int numClasses, double[] lambdas, double alpha,
                                         List<Integer> classes, List<String> messages, List<String> testMessages) {
        int n = messages.size();
        double[] classesFrequency = new double[numClasses];
        double[] totalWords = new double[numClasses];
        List<HashMap<String, Integer>> dicts = new ArrayList<>();
        for (int i = 0; i < numClasses; i++) {
            dicts.add(new HashMap<>());
        }
        for (int i = 0; i < n; i++) {
            int cl = classes.get(i);
            classesFrequency[cl]++;
            HashSet<String> words = new HashSet<>();
            Collections.addAll(words, messages.get(i).trim().split(" "));
            for (String word: words) dicts.get(cl).merge(word, 1, (x, y) -> x + y);
        }
        for (int i = 0; i < numClasses; i++) {
            totalWords[i] = dicts.get(i).size();
        }
        int m = testMessages.size();
        double[] logs = new double[numClasses];
        List<Integer> results = new ArrayList<>();
        for (String message : testMessages) {
            HashSet<String> words = new HashSet<>();
            Collections.addAll(words, message.trim().split(" "));
            int res = -1;
            for (int j = 0; j < numClasses; j++) {
                double total = classesFrequency[j] + alpha * totalWords[j];
                if (classesFrequency[j] == 0) {
                    continue;
                }
                logs[j] = Math.log(classesFrequency[j] * lambdas[j]);
                for (String word : words) {
                    double freq = dicts.get(j).getOrDefault(word, 0) + alpha;
                    logs[j] += Math.log(freq / total);
                }
                if (res == -1 || logs[res] < logs[j]) {
                    res = j;
                }
            }
            results.add(res);
        }
        return results;
    }

    public static void main(String[] args) throws IOException {
        BufferedWriter log = new BufferedWriter(new FileWriter("log-bayes.txt"));
        List<List<String>> messages = new ArrayList<>();
        List<List<Integer>> classes = new ArrayList<>();
        for (int i = 1; i < 11; i++) {
            Stream<Path> paths = Files.walk(Paths.get(System.getProperty("user.dir") + "/src/main/resources/part" + i))
                    .filter(Files::isRegularFile);
            classes.add(paths.map(path -> {
                if (path.getFileName().toString().contains("legit")) {
                    return 1;
                } else return 0;
            }).collect(Collectors.toList()));
            paths = Files.walk(Paths.get(System.getProperty("user.dir") + "/src/main/resources/part" + i))
                    .filter(Files::isRegularFile);
            messages.add(paths.map(path -> {
                try {
                    BufferedReader r = new BufferedReader(new FileReader(path.toFile()));
                    return r.lines().collect(Collectors.joining(" ")).replaceAll(" +", " ")
                            .replaceAll("[a-zA-Z:]+", "").trim();
                } catch (FileNotFoundException e) {
                    return "";
                }
            }).collect(Collectors.toList()));
        }
        double[] lambdas = new double[2];
        lambdas[0] = lambdas[1] = 1;
        XYSeriesCollection ds1 = new XYSeriesCollection();
        XYSeries series1 = new XYSeries("");
        for (double alpha = 0.01; alpha <= 1; alpha += 0.01) {
            int[][] conf = new int[2][2];
            for (int i = 0; i < 10; i++) {
                List<Integer> results = classify(2, lambdas, alpha, Utils.mergeExcept(classes, i),
                        Utils.mergeExcept(messages, i), messages.get(i));
                for (int j = 0; j < results.size(); j++) {
                    conf[results.get(j)][classes.get(i).get(j)]++;
                }
            }
            double f1 = Utils.fmeasure(conf);
            series1.add(alpha, f1);
            log.write(String.format("Alpha = %.2f, f1 = %.6f\n", alpha, f1));
            log.flush();
        }
        ds1.addSeries(series1);
        Utils.writeChartForDS("Alpha to f1", ds1, "Alpha", "F1");
        XYSeriesCollection ds = new XYSeriesCollection();
        XYSeries series = new XYSeries("");
        while (true) {
            int[][] conf = new int[2][2];
            for (int i = 0; i < 10; i++) {
                List<Integer> results = classify(2, lambdas, 0.00001, Utils.mergeExcept(classes, i),
                        Utils.mergeExcept(messages, i), messages.get(i));
                for (int j = 0; j < results.size(); j++) {
                    conf[results.get(j)][classes.get(i).get(j)]++;
                }
            }
            double f1 = Utils.fmeasure(conf);
            log.write(String.format("Lambdas: %.1f %.1f F1: %.6f\n", lambdas[0], lambdas[1], f1));
            log.write(String.format("Matrix: %d %d %d %d\n", conf[0][0], conf[0][1], conf[1][0], conf[1][1]));
            log.flush();
            series.add(Math.log10(lambdas[1]), f1);
            if (conf[0][1] == 0) break;
            else lambdas[1] *= 10;
        }
        ds.addSeries(series);
        Utils.writeChartForDS("Lambda for classifying as non-spam to f1", ds, "log10(Lambdas[1])", "F1");
    }

}
