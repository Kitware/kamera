//
// Created by Michael McDermott on 1/21/21.
//

#ifndef ROSKV_ENVOY_H
#define ROSKV_ENVOY_H


#include <iostream>
#include <fstream>
#include <string>
#include <fmt/core.h>
#include <boost/filesystem.hpp>
#include <boost/regex.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <nlohmann/json.hpp>
#include <sw/redis++/redis++.h>

using namespace sw;
namespace fs = boost::filesystem;
namespace json = nlohmann;


namespace RedisHelper {
    /** \brief get a dictionary from a Redis keyspace, optionally removing the `prefix_len` starting characters   */
    json::json get_dict(redis::Redis &rc, std::vector<std::string> &keys, std::size_t prefix_len, bool strip);

    /**
     * \brief get a dictionary from a Redis keyspace.
     * @param rc redis client
     * @param keys keys to extract
     * @param root prefix of keys
     * @return de-nested dict
     */
    json::json get_dict(redis::Redis &rc, std::vector<std::string> &keys, const std::string &root);

    /**
     * \brief get a dictionary from a Redis keyspace. If _key lacks a trailing /, it will be added
     * @param rc - redis client
     * @param key - _key/prefix of the keyspace
     * @return de-nested dict
     */
    json::json get_dict(redis::Redis &rc, const std::string &key);

    json::json put_dict(redis::Redis &rc, const std::string &key, const json::json &val);

    std::string get_redis_uri();
    void assert_redis(redis::Redis &rc);
    void assert_redis(const std::string &redis_uri);

    void assert_is_not_null(const std::string &key, const json::json &val);

};

class RedisEnvoyOpts {
public:
    static RedisEnvoyOpts from_env(const std::string &client_name);
//    RedisEnvoyOpts();
    RedisEnvoyOpts(const RedisEnvoyOpts& opts);
    RedisEnvoyOpts(const std::string &client_name,
                   const std::string &redis_uri,
                   const std::string &keypath="/agent/default");

    const std::string client_name;
    const std::string redis_uri;
    const std::string keypath;
};

std::ostream& operator<<(std::ostream &os, const RedisEnvoyOpts &opts);


/**
 * \brief Bridge to Redis with health checks
 */
class RedisEnvoy {
public:
    static RedisEnvoy from_env();


/// rule of 5
    RedisEnvoy(const RedisEnvoyOpts &opts);

    RedisEnvoy(const std::string &client_name,
               const std::string &redis_uri,
               const std::string &keypath="/agent/default");

    /// @brief `RedisEnvoy` is not copyable.
    RedisEnvoy(const RedisEnvoy &) = delete;
    RedisEnvoy& operator=(const RedisEnvoy &) = delete;

    /// @brief `RedisEnvoy` is movable.
    RedisEnvoy(RedisEnvoy &&) = default;
    RedisEnvoy& operator=(RedisEnvoy &&) = default;

    ~RedisEnvoy();

    /// this is just for very limited access. if you need more, create a separate redis client
    std::string get(const std::string &key);
    bool put(const std::string &key, const std::string &val);
    json::json get_dict(const std::string &key);
    json::json put_dict(const std::string &key, const json::json &val);
    json::json del(const std::string &key);
    std::string client_name();
    std::string echo(const std::string &msg);


protected:
    void register_();
    void deregister_();
    const std::string client_name_;
    const std::string redis_uri;
    const std::string keypath_;
    redis::Redis rc;
    json::json _kwargs;

};

namespace EnvoyHelper {
    /** initialize entirely from ENV. This facilitates static init */
    RedisEnvoy from_env();
}


#endif //ROSKV_ENVOY_H
