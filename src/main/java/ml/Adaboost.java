package ml;

public class Adaboost {

//    public static void boostSet(String fileName, int steps) throws IOException {
//        String[] tokens = fileName.split("/");
//        String name = tokens[tokens.length - 1].split("\\.")[0];
//        BufferedWriter log = new BufferedWriter(new FileWriter("log-" + name + ".txt"));
//        log.write(steps + "\n");
//        List<String[]> values = new CSVReader(new FileReader(fileName)).readAll();
//        values.remove(0);  // header
//        int samples = values.size();
//        int attributes = values.get(0).length - 1;
//        double[][] objects = new double[samples][attributes];
//        int[] classes = new int[samples];
//        for (int i = 0; i < samples; i++) {
//            for (int j = 0; j < attributes; j++) {
//                objects[i][j] = Double.valueOf(values.get(i)[j]);
//            }
//            classes[i] = values.get(i)[attributes].equals("P") ? 1 : -1;
//        }
//        double[][] kernel = new double[samples][samples];
//        for (int i = 0; i < samples; i++) {
//            for (int j = 0; j < samples; j++) {
//                kernel[i][j] = classes[i] * classes[j] * Utils.scalarProduct(objects[i], objects[j]);
//            }
//        }
//        double[] weights = new double[samples];
//        double[]
//        for (int i = 0; i < samples; i++) {
//            weights[i] = 1D / samples;
//        }
//        for (int i = 0; i < steps; i++) {
//
//        }
//    }
//
//    public static void main(String[] args) throws IOException {
//        boostSet("src/main/resources/chips.csv", 20);
//        boostSet("src/main/resources/geyser.csv", 20);
//    }

}
