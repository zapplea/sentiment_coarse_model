import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import edu.stanford.nlp.process.DocumentPreprocessor; //You should copy stanford-english-corenlp-2018-02-27-models/edu to this diretory or add path to library.
import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.trees.*;
import edu.stanford.nlp.parser.lexparser.LexicalizedParser;

public class StanfordParser {
    private StanfordParser() {} // static methods only
    public static void main(String[] args) {

        String parserModel = "edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz";
        LexicalizedParser lp = LexicalizedParser.loadModel(parserModel);

        //String textFile = "text.txt";
        String textFile = "semeval_restaurant.txt";
        String filePath = "/Users/lujunyu/IdeaProjects/data.json";

        CreateFileUtil cf = new CreateFileUtil();
        cf.createJsonFile(filePath);

        DP(lp, textFile,cf,filePath);

    }

    public static void DP(LexicalizedParser lp, String filename,CreateFileUtil cf,String filePath) {

        TreebankLanguagePack tlp = lp.treebankLanguagePack(); // a PennTreebankLanguagePack for English  
        GrammaticalStructureFactory gsf = null;
        if (tlp.supportsGrammaticalStructures()) {
            gsf = tlp.grammaticalStructureFactory();
        }

        StringBuilder SB = new StringBuilder();

        SB.append("{\n");
        int i = 0;

        for (List<HasWord> sentence : new DocumentPreprocessor(filename)) {

            Tree parse = lp.apply(sentence);
            //parse.pennPrint();
            //System.out.println();

            if (gsf != null) {
                GrammaticalStructure gs = gsf.newGrammaticalStructure(parse);
                Collection tdl = gs.typedDependenciesCCprocessed();
                Iterator iter = tdl.iterator();
                SB.append("\""+i+"\":\"");
                i++;

                while(iter.hasNext()){
                    SB.append(iter.next().toString()+";");
                    //System.out.println(data);

                }

                SB.append("\",\n");

                //System.out.println(tdl);  
                //System.out.println();  
            }
            System.out.println(i);
        }
        cf.append(SB.subSequence(0, SB.length()-2).toString()+"\n}",filePath);
    }


}