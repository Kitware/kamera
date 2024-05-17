
#include <iostream>
#include <regex>
#include <memory>
#include <nlohmann/json.hpp>
#include <sw/redis++/redis++.h>
#include <roskv/roskv.h>
#include <roskv/envoy.h>
#include <roskv/archiver.h>
#include <ros/ros.h>

// No Color
#define NC "\033[0m"
#define BLU "\033[0;34m"
#define GRN "\033[0;32m"
#define RED "\033[0;31m"
#define PUR "\033[0;35m"
#define ROS_GREEN(mystr) ROS_INFO(GRN mystr NC)
#define ROS_BLUE(mystr) ROS_INFO(BLU mystr NC)

using namespace std;
namespace json = nlohmann;
using namespace sw;

json::json nop(json::json const & j) {
    return j.unflatten();
}

std::vector<std::string> split(const std::string str, const std::string regex_str)
{
    std::regex regexz(regex_str);
    std::vector<std::string> list(std::sregex_token_iterator(str.begin(), str.end(), regexz, -1),
                                  std::sregex_token_iterator());
    return list;
}

json::json read_jsonl(std::string const &s) {
    size_t idx = 0;
    std::string token;
//    std::vector<json::json> out;
    json::json out;
    auto lines = split(s, "\n");
    for (auto & line : lines) {
//        cout <<  ". : " << line << endl;
        try {
            out[idx++] = json::json::parse(line);
        } catch (json::detail::parse_error&) {
            cerr << "Not able to parse: " << line << endl;
        }
    }

    return out;
}

void TestMove2(std::unique_ptr<RedisEnvoy> envoy) {
    cout << "TestMove" << envoy->client_name() << endl;
}

void TestMove(std::unique_ptr<RedisEnvoy> envoy) {
    TestMove2(std::move(envoy));
}

class Scope {
public:
    Scope() {}
    void onInit() {
        ROS_GREEN("test_envoy Scope::onInit()");
        RedisEnvoyOpts re_opts = RedisEnvoyOpts::from_env();
        string redis_addr = RedisHelper::get_redis_uri();

        p_envoy = std::make_shared<RedisEnvoy>("testbed", redis_addr, opts.key);
        p_envoy = std::make_shared<RedisEnvoy>("testbed", redis_addr, opts.key);

        cout << " << end init " << p_envoy->client_name() << endl;
        //p_envoy->set("foo", "bar");
    }
    void test() {
        ROS_BLUE("test_envoy Scope::test()");
        cout << "test " << p_envoy->echo("hello envoy test 1")   << endl;

        long int sec = 1611595445;
        long int nsec = 123456789;
        json::json kwargs = p_envoy->get_dict(opts.key);
        cout << "kwargs: " << kwargs.dump() << endl;
//        TestMove(std::move(p_envoy));
        string filename = ArchiverHelper::generateFilename(p_envoy, opts, sec, nsec);
        cout << "filename: " << filename << endl;
    }

    ArchiverOpts opts = ArchiverOpts::from_env();
private:
    std::shared_ptr<RedisEnvoy> p_envoy;
};

bool test_redis() {
    std::string hostname = "127.0.0.1";
    std::string port = "6379";
    if(const char* env_host = std::getenv("REDIS_HOST")) {
        hostname = std::string{env_host};
    }
    if(const char* env_port = std::getenv("REDIS_PORT")) {
        port = std::string{env_port};
    }
    auto url = "tcp://" + hostname + ":" + port;

    auto rc = redis::Redis(url);
    std::string key = "foo";
    redis::OptionalString val = rc.get(key);    // val is of type OptionalString. See 'API Reference' section for details.
    if (bool(val)) {
        // Dereference val to get the returned value of std::string type.
        std::cout << key << ": " << *val << std::endl;
    } else {
        std::cerr << "Key not found: " << key << std::endl;
    }
    auto valu = val.value();
    cout << valu << endl;
}

bool test_json() {
        json::json j, k;
        j["pi"] = 3.141;
        j["answer"]["everything"] = 42;
        j["answer"]["nothing"] = "pain";
        k[0] = "foo";
        auto answer = j["answer"];
        string jraw = "{ \"happy\": true, \"pi\": 3.141 }\n{ \"things\": false, \"e\": 2.7}\n{dead}\n{\"end\": null}";
//    auto j3 = json::parse("{}");
        auto out = read_jsonl(jraw);

//    cout << j.dump() << endl;
//    cout << k.dump() << endl;
        cout << out.dump() << endl;
//    cout << answer.dump() << endl;
//    cout << j.flatten().dump() << endl;
//    cout << j3.dump() << endl;
}

bool test_roskv() {
    hello_roskv();
    string redis_addr = RedisHelper::get_redis_uri();
    RedisHelper::assert_redis(redis_addr);
    ArchiverFormatter af{redis_addr, "/sys/arch", "center", "rgb"};
    af.fetchKwargs();
    cerr << af.kwargs().dump() << endl;
    cout << af.generateFilename("jpg") << endl;
    ROS_GREEN("completed test_roskv");
}

bool test_with_env() {
    const char *env_a = std::getenv("TEST_ENV");
    if (!(env_a && env_a[0])) {
        cerr << "Skipping test_with_env" << endl;
        return false;
    }

    std::unique_ptr<ArchiverFormatter> archiver_;
    archiver_ = std::make_unique<ArchiverFormatter>(ArchiverFormatter::from_env());
    archiver_->fetchKwargs();
    cout << archiver_->generateFilename("jpg") << endl;
    ROS_GREEN("completed test_with_env");
}

bool test_envoy() {
    ROS_BLUE("start> test_envoy");
    Scope scope;
    ROS_BLUE("start> test_envoy::onInit");
    scope.onInit();
    ROS_BLUE("start> test_envoy::test");
    scope.test();
    ROS_GREEN("completed test_envoy");
}

int main(int argc,char** argv) {
    ros::init(argc, argv, "test_roskv");
    ros::start();
    ros::NodeHandle nh;
    auto end_ = nh.createTimer(ros::Duration(1.0),
        [](const ros::TimerEvent &event) {
            ROS_GREEN("complete!");
            ros::shutdown();
        }, true, true);
    ROS_INFO("ros started");
    test_json();
    test_redis();
    test_roskv();
    test_with_env();
    test_envoy();
    ros::spin();
    std::cout << "clean exit" << std::endl;
    return 0;
}