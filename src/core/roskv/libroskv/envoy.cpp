#include <fstream>
#include <vector>
#include <fmt/args.h>
#include "roskv/envoy.h"


namespace RedisHelper {
    json::json get_dict(redis::Redis &rc, std::vector<std::string> &keys, const std::size_t prefix_len, bool strip) {
        json::json out;
        std::vector<redis::OptionalString> res;
        rc.mget(keys.begin(), keys.end(), std::back_inserter(res));
        assert(keys.size() == res.size());
        auto count = 0;
        for (auto &k : keys) {
            if (strip) {
                k = k.erase(0, prefix_len);
            }
            out[k] = *res[count];
            count++;
        }
        return out;
    }

    json::json get_dict(redis::Redis &rc, std::vector<std::string> &keys, const std::string &root) {
        for (auto &k : keys) {
            k = root + k;
        }
        return get_dict(rc, keys, root.size(), true);
    }

    json::json get_dict(redis::Redis &rc, const std::string &key) {
        json::json tmp;
        auto cursor = 0LL;
        auto key_ = key;
        if (key_.back() != '/') {
            key_ = key_ + '/';
        }
        std::vector<std::string> keys;
        while (true) {
            cursor = rc.scan(cursor, key_ + '*', 20, std::inserter(keys, keys.begin()));
            if (cursor == 0) {
                break;
            }
        }
        return get_dict(rc, keys, key_.size(), true);
    }

    json::json put_dict(redis::Redis &rc, const std::string &key, const json::json &val) {
        json::json out;
        if (val.is_primitive()) {
            rc.set(key, std::string(val));
            out = true;
            return out;
        }
        auto key_ = key;
        if (key_.back() != '/') {
            key_ = key_ + '/';
        }
        auto tmp = val.flatten();
        for (auto &el: tmp.items()) {
            auto k = key_ + el.key();
            out[k] = rc.set(k, std::string(el.value()));
        }
        return out;
    }

    std::string get_redis_uri() {

        const char *env_a = std::getenv("REDIS_ADDR");
        if (env_a && env_a[0]) {
            return std::string(env_a);
        }
        const char *env_schm = std::getenv("REDIS_SCHEME");
        const char *env_port = std::getenv("REDIS_PORT");
        const char *env_host = std::getenv("REDIS_HOST");

        std::string scheme, port, host;
        if (env_schm && env_schm[0]) {
            scheme = env_schm;
        } else {
            scheme = "tcp";
        }

        if (env_port && env_port[0]) {
            port = env_port;
        } else {
            port = "6379";
        }
        /// verify port is a number
        try {
            int x = std::stoi(port, nullptr, 10);
            if ((x < 0) || (x > 65535)) {
                throw std::invalid_argument("port is out of range");
            }
        } catch (std::invalid_argument &e) {
            std::cerr << "ENV variable REDIS_PORT must be a valid port, got " << port << std::endl;
            throw e;
        }


        if (env_host && env_host[0]) {
            host = env_host;
        } else {
            host = "127.0.0.1";
        }
        return scheme + "://" + host + ":" + port;

    }

    void assert_redis(redis::Redis &rc) {
        std::string msg = "test123";
        auto out = rc.echo(msg);
        if (out != msg) {
            throw redis::IoError("Echo response failed");
        }
    }
    void assert_redis(const std::string &redis_uri) {
        try {
            auto rc = redis::Redis(redis_uri);
            assert_redis(rc);
        } catch (redis::IoError &e) {
            std::cerr << "Failed to connect to redis at: " << redis_uri << std::endl;
            throw;
        }
    }
    void assert_is_not_null(const std::string &key, const json::json &val) {
        if (val.is_null()) {
            std::string msg = "Assert fail: Redis Key " + key + " must not be null";
            throw std::invalid_argument(msg);
        }
    }
}

RedisEnvoyOpts RedisEnvoyOpts::from_env(const std::string &client_name) {
    const char *env_cfg_file = std::getenv("ENVOY_CFG_FILE");
    const char *env_cfg = std::getenv("ENVOY_CFG");

    json::json cfg;
    std::string scfg;
    std::string arch_key;
    if (env_cfg_file && env_cfg_file[0]) {
        boost::filesystem::path cfg_path{env_cfg_file};
        if (!boost::filesystem::exists(cfg_path)) {
            std::cerr << "Warning: specified ENVOY_CFG_PATH but does not exist: " << cfg_path << std::endl;
        } else {
            std::ifstream ifs(cfg_path.string());
            cfg = json::json::parse(ifs);
        }
    }
    if (env_cfg && env_cfg[0]) {
        scfg = std::string{env_cfg};
        json::json ecfg = json::json::parse(scfg);
        cfg.update(ecfg);
    }

    std::string redis_uri;
    if (cfg["redis_uri"].is_null()) {
        redis_uri = RedisHelper::get_redis_uri();
    } else {
        redis_uri = std::string(cfg["redis_uri"]);
    }

    return RedisEnvoyOpts{client_name, redis_uri, "/agent/default"};
}

RedisEnvoyOpts::RedisEnvoyOpts(const RedisEnvoyOpts& opts) = default;


//RedisEnvoyOpts::RedisEnvoyOpts() :
//        client_name{"default_client"},
//        redis_uri{"tcp://localhost:6379"},
//        keypath{"/agent/default"} {}

RedisEnvoyOpts::RedisEnvoyOpts(const std::string &client_name,
                       const std::string &redis_uri,
                       const std::string &keypath) :
        client_name{client_name},
        redis_uri{redis_uri},
        keypath{keypath} {}

std::ostream& operator<<(std::ostream &os, const RedisEnvoyOpts &opts) {
    os << "RedisEnvoyOpts{" << opts.client_name << ", " << opts.redis_uri << ", " << opts.keypath << "}";
    return os;
}

RedisEnvoy::RedisEnvoy(const RedisEnvoyOpts &opts) :
    client_name_{opts.client_name},
            rc{opts.redis_uri},
            keypath_{opts.keypath} {
std::cerr << "RedisEnvoy " << opts.client_name << " created at " << opts.redis_uri << std::endl;

}


RedisEnvoy::RedisEnvoy(const std::string &client_name,
                       const std::string &redis_uri,
                       const std::string &keypath) :
        client_name_{client_name},
        rc{redis_uri},
        keypath_{keypath} {
    std::cerr << "DEPRECATED RedisEnvoy " << client_name << " created at " << redis_uri;
}

RedisEnvoy::~RedisEnvoy() {
//    logr << "~Envoy dtor" << endl;
//    rc.~Redis();
}

RedisEnvoy RedisEnvoy::from_env() {
    return EnvoyHelper::from_env();
}


std::string RedisEnvoy::get(const std::string &key) {
    auto tmp = rc.get(key);
    if (bool(tmp)) {
        return tmp.value();
    }
    std::string msg = "get failed. Redis Key " + key + " must not be null";
    throw std::invalid_argument(msg);
}
bool RedisEnvoy::put(const std::string &key, const std::string &val) {
    return rc.set(key, val);
}
json::json RedisEnvoy::get_dict(const std::string &key) {
    return RedisHelper::get_dict(rc, key);
}
json::json RedisEnvoy::put_dict(const std::string &key, const json::json &val) {
    return RedisHelper::put_dict(rc, key, val);
}

std::string RedisEnvoy::client_name() {
    return std::string(client_name_);
}
std::string RedisEnvoy::echo(const std::string &msg) {
    auto out = rc.echo(msg);
    if (out != msg) {
        throw redis::IoError("Echo response failed");
    }
    return out;
}

void RedisEnvoy::register_() {

}
void RedisEnvoy::deregister_() {

}


namespace EnvoyHelper {
    RedisEnvoy from_env() {
        const char *env_name = std::getenv("ENVOY_NAME");
        const char *env_cfg = std::getenv("ENVOY_CFG");
        std::string client_name;
        std::string scfg;
        if (!(env_cfg && env_cfg[0])) {
            scfg = "";
//            throw std::invalid_argument("ENV variable ENVOY_CFG must be specified");
        } else {
            scfg = std::string{env_cfg};
        }
        if (!(env_name && env_name[0])) {
            client_name = "nemo_cpp";
        } else {
            client_name = std::string{env_name};
        }

        const std::string redis_uri = RedisHelper::get_redis_uri();
        std::cerr << "Establishing redis connection " << client_name << " on " << redis_uri << std::endl;
        return RedisEnvoy{client_name, redis_uri};
    }
}
