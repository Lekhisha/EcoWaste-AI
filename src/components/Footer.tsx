
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faFacebookF } from "@fortawesome/free-brands-svg-icons";
import { faInstagram } from "@fortawesome/free-brands-svg-icons";
import { faXTwitter } from "@fortawesome/free-brands-svg-icons";
const Footer = () => {
  return (
    <>
    <div className="relative py-14  bg-gradient-to-r from-amber-200 to-green-500 rounded-t-4xl">
        <div className="container py-4 ">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 border-b-2 border-cyan-950 pb-6">
            <div>
              <div className="text-3xl font-bold text-green-600">
                Eco<span className="text-cyan-950">-Waste AI</span>
              </div>
              <p className="py-4 font-semibold text-cyan-950">
                Help us make a better future and connect with us.{" "}
              </p>

              <div className="flex gap-4 text-2xl">
                <div>
                  <FontAwesomeIcon
                    icon={faFacebookF}
                    className="hover:text-blue-500"
                  />
                </div>
                <div>
                  <FontAwesomeIcon
                    icon={faInstagram}
                    className="hover:text-purple-600"
                  />
                </div>
                <div>
                  <FontAwesomeIcon
                    icon={faXTwitter}
                    className="hover:text-blue-500"
                  />
                </div>
              </div>
            </div>

            <div className="flex gap-5">
              <div>
                <div className="text-cyan-950 text-2xl font-semibold">
                  Resources
                </div>
                <ul className="font-semibold text-cyan-950">
                  <li>Blog</li>
                  <li>Help Center</li>
                  <li>FAQ</li>
                </ul>
              </div>

              <div>
                <div className="text-cyan-950 text-2xl font-semibold">
                  Legal
                </div>
                <ul className="text-cyan-950 font-semibold">
                  <li>Terms of Service</li>
                  <li>Privacy Policy</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  )
}

export default Footer