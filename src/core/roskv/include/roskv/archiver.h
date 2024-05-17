//
// Created by Michael McDermott on 1/21/21.
//

#ifndef ROSKV_ARCHIVER_H
#define ROSKV_ARCHIVER_H


#include <iostream>
#include <string>
#include <fmt/core.h>
#include <boost/filesystem.hpp>
#include <boost/regex.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <nlohmann/json.hpp>
#include <sw/redis++/redis++.h>

inline std::string isoformat_u(time_t tv_sec, long tv_nsec);

using namespace sw;
namespace fs = boost::filesystem;
namespace json = nlohmann;

/** Gets attributes (modality/fov) about a camera from the environment */
class Cameratype {
public:

    Cameratype(const std::string &fov, const std::string &mode);
    static Cameratype from_env();
    const std::string fov; /// Field of view
    const std::string mode; /// Modality e.g. color/uv
};

namespace FormatterHelper {
    /** \brief Strip out any _kwargs not in the format string, and add placeholders as needed */
    json::json conformKwargsToFormatter(const std::string &fmtr, const json::json &kwargs);

    std::string applyFormat(const std::string &fmtr, const json::json &kwargs, bool partial);

    inline std::string applyFormat(const std::string &fmtr, const json::json &kwargs);



    std::string generateFilename(RedisEnvoy &envoy, const std::string &ext);
//        fetchArchiveKwargs(_key);
//        std::string fmtr = _kwargs["template"];
//        std::string stime = boost::posix_time::to_iso_string(time);
//        stime[8] = '_';
//        _kwargs["time"] = stime;
//        _kwargs["ext"] = ext;
//        auto tmp = FormatterHelper::conformKwargsToFormatter(fmtr, _kwargs);
//        return FormatterHelper::applyFormat(fmtr, tmp);

};

class ArchiverOpts {
public:
    ArchiverOpts(const std::string &key, const std::string &cam_fov, const std::string &mode);
    static ArchiverOpts from_env();

    const std::string key;
    const std::string cam_fov;
    const std::string mode;
};

namespace ArchiverHelper {
    std::string generateFilename(std::shared_ptr<RedisEnvoy> envoy, const ArchiverOpts &opts,
                                 long int sec, long int nsec);
    json::json fetchArchiveKwargs(std::shared_ptr<RedisEnvoy> envoy, const ArchiverOpts &opts);

    int get_is_archiving(std::shared_ptr<RedisEnvoy> envoy, const std::string &key);

}

/**
 * \brief Formats filenames based on attributes stored in Redis
 */
class ArchiverFormatter {
public:
    ArchiverFormatter(const std::string &redis_host, const std::string &redis_port,
                      const std::string &key, const std::string &cam_fov, const std::string &mode);

    ArchiverFormatter(const std::string &redis_uri,
                      const std::string &key, const std::string &cam_fov, const std::string &mode);

    static ArchiverFormatter from_env();

    static json::json fetchArchiveKwargs(redis::Redis &rc, const std::string &key, const std::string &cam_fov, const std::string &mode);
    /** \brief Fetch keys from Redis keypath and update _kwargs accordingly */
    bool fetchKwargs();

    /** \brief Emit a filename based on the timestamp */
    std::string generateFilename(const boost::posix_time::ptime &time, const std::string &ext);
    std::string generateFilename(const std::string &ext);

    json::json kwargs();


protected:
    redis::Redis rc;
    std::string _key;
    json::json _kwargs;
    std::string cam_fov;
    std::string mode;
};


#endif //ROSKV_ARCHIVER_H
