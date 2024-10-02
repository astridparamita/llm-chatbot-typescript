import { ChatOpenAI } from "@langchain/openai";
import { OpenAIEmbeddings } from "@langchain/openai";
import initAgent from "./agent";
import { initGraph } from "../graph";
import { sleep } from "@/utils";
// import initGenerateAnswerChain from "./chains/answer-generation.chain"
// import initRephraseChain from "./chains/rephrase-question.chain";
import initGenerateAuthoritativeAnswerChain from "./chains/authoritative-answer-generation.chain";
import { ChatbotResponse } from "./history";

// tag::call[]
export async function call(input: string, sessionId: string): Promise<string> {
  const llm = new ChatOpenAI({
    openAIApiKey: process.env.OPENAI_API_KEY,
    // Note: only provide a baseURL when using the GraphAcademy Proxy
    configuration: {
      baseURL: process.env.OPENAI_API_BASE,
    },
  });
  const embeddings = new OpenAIEmbeddings({
    openAIApiKey: process.env.OPENAI_API_KEY,
    configuration: {
      baseURL: process.env.OPENAI_API_BASE,
    },
  });
  // Get Graph Singleton
  const graph = await initGraph();

  const agent = await initAgent(llm, embeddings, graph);
  const res = await agent.invoke({ input }, { configurable: { sessionId } });

  return res;
}
// export async function call(input: string, sessionId: string): Promise<string> {
//   // TODO: Replace this code with an agent 
//   // await sleep(2000);
//   // return input;

//   const llm = new ChatOpenAI() // Or the LLM of your choice
// const answerChain = initGenerateAuthoritativeAnswerChain(llm)

// const output = await answerChain.invoke({
//   question: 'Who is the CEO of Neo4j?',
//   context: 'Neo4j CEO: Emil Eifrem',
// }) // Emil Eifrem is the CEO of Neo4j

//   // const llm = new ChatOpenAI() // Or the LLM of your choice
//   // const rephraseAnswerChain = initRephraseChain(llm)
//   // const history = [
//   //   {
//   //     input: "Can you recommend me a film?",
//   //     output: "Sure, I recommend The Matrix",
//   //   } as ChatbotResponse,
//   // ];  
//   // const output = await rephraseAnswerChain.invoke({
//   //   input,
//   //   history ,
//   // }) // Other than Toy Story, what movies has Tom Hanks acted in?  
//   return output

  
// }
// end::call[]
