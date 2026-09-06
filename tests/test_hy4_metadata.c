/* Vocabulary and chat regression for Hy4-preview-Flash-STQ1_0.
 * Expected IDs were generated with llama.cpp 34cccef vocabulary-only APIs;
 * chat fixtures use the supplied GGUF's tokenizer.chat_template verbatim.
 * No GPU context is created and no model tensor data is read.
 * Usage: test_hy4_metadata /path/to/model-dense-f16head.gguf
 */
#include "../ds4.c"
#include <assert.h>
static int failures;
static void expect_ids(const char *name,const token_vec *got,const int *want,int n) {
    if(got->len!=n || memcmp(got->v,want,(size_t)n*sizeof(int))) {
        fprintf(stderr,"FAIL %s: got %d tokens, expected %d\n",name,got->len,n);
        for(int i=0;i<got->len;i++)fprintf(stderr,"%s%d",i?",":"",got->v[i]);
        fputc('\n',stderr); failures++;
    } else printf("PASS %s (%d tokens)\n",name,n);
}
int main(int argc,char **argv) {
    if(argc!=2) {fprintf(stderr,"usage: %s HY4_MODEL.gguf\n",argv[0]);return 2;}
    ds4_engine e={0};
    model_open(&e.model,argv[1],false,false);
    config_validate_model(&e.model);
    assert(DS4_MODEL_VARIANT==DS4_VARIANT_HY4);
    assert(DS4_N_LAYER==78 && DS4_N_EMBD==6144 && DS4_N_EXPERT==256 && DS4_N_EXPERT_USED==8);
    vocab_load(&e.vocab,&e.model);
    assert(e.vocab.hy4_tokenizer && e.vocab.n_vocab==120832);
    assert(e.vocab.bos_id==120000 && e.vocab.eos_id==120025);
    assert(flash_moe_quant_type_id("STQ1_0", NULL)==DS4_TENSOR_STQ1_0);
    assert(routed_expert_block_bytes(DS4_TENSOR_STQ1_0)==42);
    {
        static const int expected[] = {12433,0,356,3283,259,14584,30645,491};
        token_vec got={0};
        tokenize_rendered_chat_vocab(&e.vocab,"Hello! I'm a coding assistant.\n",&got);
        expect_ids("english",&got,expected,sizeof(expected)/sizeof(expected[0]));
        token_vec_free(&got);
    }
    {
        static const int expected[] = {2345,1147,2533,11,389,3894,326,1315,1142,803,389,198,2,1702,28,6636,17243,29041,15,198};
        token_vec got={0};
        tokenize_rendered_chat_vocab(&e.vocab,"def add(x, y):\n    return x + y\n# value=1234567890\n",&got);
        expect_ids("code",&got,expected,sizeof(expected)/sizeof(expected[0]));
        token_vec_free(&got);
    }
    {
        static const int expected[] = {17134,298,3913,1855,10629,4625,320,96698,2625,74420,7144,198};
        token_vec got={0};
        tokenize_rendered_chat_vocab(&e.vocab,"你好，世界！中文测试。日本語と한국어\n",&got);
        expect_ids("cjk",&got,expected,sizeof(expected)/sizeof(expected[0]));
        token_vec_free(&got);
    }
    {
        static const int expected[] = {220,259,197,50623,268,271,256,201,198};
        token_vec got={0};
        tokenize_rendered_chat_vocab(&e.vocab,"  a\t\tb\n\n c  \r\n",&got);
        expect_ids("whitespace",&got,expected,sizeof(expected)/sizeof(expected[0]));
        token_vec_free(&got);
    }
    {
        static const int expected[] = {66,2592,729,112325,96542,222,24679,102,23429,25391,119,319,20639,2227,25787,220,983,21};
        token_vec got={0};
        tokenize_rendered_chat_vocab(&e.vocab,"café naïve 🚀👩‍💻 é — © 2026",&got);
        expect_ids("emoji",&got,expected,sizeof(expected)/sizeof(expected[0]));
        token_vec_free(&got);
    }
    {
        static const int expected[] = {6636,17243,29041,16970,19894,33163,2999,220,16,13,1069,16209,52864,19,220,15,87,21738,40905};
        token_vec got={0};
        tokenize_rendered_chat_vocab(&e.vocab,"12345678901234567890 1.25-bit HY4 0xABC_def",&got);
        expect_ids("numbers",&got,expected,sizeof(expected)/sizeof(expected[0]));
        token_vec_free(&got);
    }
    {
        static const int expected[] = {120000,611,10372,120001,120029,120030,120035};
        token_vec got={0};
        tokenize_rendered_chat_vocab(&e.vocab,"<｜hy_start:6124c78e｜>assistant<｜hy_middle:6124c78e｜><think:6124c78e></think:6124c78e><tool_calls:6124c78e>",&got);
        expect_ids("controls",&got,expected,sizeof(expected)/sizeof(expected[0]));
        token_vec_free(&got);
    }
    {
        static const int expected[] = {120000,13251,120001,6043,41174,13,120039,59785,287,62,4138,506,25,3202,32980,1121,120025,120000,3717,120001,12433,120025,120000,611,10372,120001,120029,120030};
        token_vec got={0};
        ds4_encode_chat_prompt(&e,"Be concise.","Hello",DS4_THINK_NONE,&got);
        expect_ids("chat_no_think",&got,expected,sizeof(expected)/sizeof(expected[0]));
        token_vec_free(&got);
    }
    {
        static const int expected[] = {120000,13251,120001,6043,41174,13,120039,59785,287,62,4138,506,25,19369,120025,120000,3717,120001,12433,120025,120000,611,10372,120001,120029};
        token_vec got={0};
        ds4_encode_chat_prompt(&e,"Be concise.","Hello",DS4_THINK_HIGH,&got);
        expect_ids("chat_high",&got,expected,sizeof(expected)/sizeof(expected[0]));
        token_vec_free(&got);
    }
    {
        static const int expected[] = {120000,13251,120001,120039,59785,287,62,4138,506,25,3202,32980,1121,120025,120000,3717,120001,12433,120025,120000,611,10372,120001,120029,120030};
        token_vec got={0};
        ds4_chat_begin(&e,&got);
        ds4_chat_append_message(&e,&got,"user","Hello");
        ds4_chat_append_assistant_prefix(&e,&got,DS4_THINK_NONE);
        expect_ids("chat_synth_system",&got,expected,sizeof(expected)/sizeof(expected[0]));
        token_vec_free(&got);
    }
    {
        static const int expected[] = {120000,13251,120001,6043,41174,13,120039,59785,287,62,4138,506,25,3202,32980,1121,120025,120000,3717,120001,12433,120025,120000,611,10372,120001,120029,120030,10648,0,120025,120000,3717,120001,58142,120025,120000,611,10372,120001,120029,120030};
        token_vec got={0};
        ds4_chat_begin(&e,&got);
        ds4_chat_append_message(&e,&got,"system","Be concise.");
        ds4_chat_append_message(&e,&got,"user","Hello");
        ds4_chat_append_message(&e,&got,"assistant","Hi!");
        ds4_chat_append_message(&e,&got,"user","Continue");
        ds4_chat_append_assistant_prefix(&e,&got,DS4_THINK_NONE);
        expect_ids("chat_history",&got,expected,sizeof(expected)/sizeof(expected[0]));
        token_vec_free(&got);
    }
    {
        static const int expected[] = {120000,13251,120001,6043,41174,13,120039,59785,287,62,4138,506,25,3202,32980,1121,120025,120000,3717,120001,8597,120025,120000,611,10372,120001,120029,120030,120035,120033,1629,120040,5096,120041,120042,87,1769,120043,120034,120036,120025,120000,23441,120001,120037,18925,198,120038,120025,120000,611,10372,120001,120029,120030};
        token_vec got={0};
        ds4_chat_begin(&e,&got);
        ds4_chat_append_message(&e,&got,"system","Be concise.");
        ds4_chat_append_message(&e,&got,"user","Read");
        ds4_chat_append_message(&e,&got,"assistant","<tool_calls:6124c78e><tool_call:6124c78e>read<arg_key:6124c78e>path</arg_key:6124c78e><arg_value:6124c78e>x.c</arg_value:6124c78e></tool_call:6124c78e></tool_calls:6124c78e>");
        ds4_chat_append_message(&e,&got,"tool","hello\n");
        ds4_chat_append_assistant_prefix(&e,&got,DS4_THINK_NONE);
        expect_ids("chat_tool",&got,expected,sizeof(expected)/sizeof(expected[0]));
        token_vec_free(&got);
    }
    vocab_free(&e.vocab);model_close(&e.model);return failures?1:0;
}
